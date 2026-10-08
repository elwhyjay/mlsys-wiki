# SGLang PD 분리 (2) - DP와 TP는 어떻게 지원되는가

> 원문: https://zhuanlan.zhihu.com/p/1921162497592886258

앞의 글들에서 SGLang의 분산 병렬 전략과 SGLang의 PD 분리 구현을 각각 다루었다. 이번 글에서는 앞의 분석들을 엮어 SGLang의 DeepSeek 재현 작업을 이야기한다. 이렇게 하면 DeepSeek의 DP, TP, EP 파라미터 설정 뒤에 있는 이유를 더 깊이 이해할 수 있다.

### MoE 모델의 병렬 이해하기

앞서 [SGLang 소스 코드 (분산 병렬)](https://zhuanlan.zhihu.com/p/1909915806906713201) 글에서 SGLang의 매우 복잡한 DP, TP, EP 혼합 병렬 방식을 분석했다. 이 방식은 주로 DeepSeek 같은 MLA+MoE 모델을 위해 설계된 것이다. 개인적인 이해를 바탕으로 이 복잡한 병렬 설정이 어디에서 나왔는지 정리해 본다. [텐센트 一念LLM 신버전 출시: 핵심 스케줄링 정면 돌파, 풀버전 DeepSeek throughput 48% 향상](https://zhuanlan.zhihu.com/p/1920496000939857868)을 참고했다.

이해에 앞서 Attention과 FFN(MoE)을 구분해 두자. Attention의 TP는 ATP, FFN의 TP는 TP라고 부른다.

예전 Dense 모델 시대에는 보통 단일 노드 안에서 TP를 했다. TP는 통신량이 크지만 메모리 사용량을 효과적으로 낮추고 latency도 줄일 수 있기 때문이다. 그리고 여러 노드에서는, 모델이 들어가기만 하면 인스턴스를 여러 개 띄우면 됐고(이것이 DP이며 통신 오버헤드가 없다), 들어가지 않으면 PP를 썼다. PP는 TP보다 통신량이 훨씬 적고 마지막 activation 값만 전달하기 때문이다.

DeepSeek이 나온 뒤 Attention은 GQA에서 MLA로, FFN은 MoE로 바뀌었다. MLA는 head가 하나뿐이라 qkv_proj weight를 TP로 분할할 수 없고, 그래서 카드마다 완전한 KV cache를 보관해야 한다. 그 결과 Attention에는 DP+ATP 혼합 병렬이 도입되었다. 여기서 ATP는 SGLang 코드 기준으로 q_proj와 $W^{UKV}$를 나누는 것이며, DP 도입에는 영향을 주지 않는다.

이 부분은 조금 더 자세히 볼 수 있다. Decode의 성능은 기본적으로 batch size의 크기가 결정하므로 KV cache의 메모리 점유가 매우 중요하다. DP를 도입하면 KV cache 저장량이 줄어들어, MoE layer를 계산할 때 메모리가 터지지 않도록 더 큰 $bs$ 예산을 확보할 수 있다. **따라서 DP가 가속하는 것은 TP 병렬 아래의 MoE layer다. 하지만 MoE에 EP를 쓴다면 all gather 자체가 전혀 필요 없고, MoE layer의 batch size는 여전히 $bs/n_{DP}$이므로 larger batch size라는 말은 성립하지 않는다.**

이상은 TP일 때의 상황이다. TP의 단점은 통신량이 커서 여러 노드로 확장하기 어렵다는 것이다. 게다가 DeepSeek v3의 intermediate size는 18432인데, TP=32이면 카드당 차원이 576이 되어 GPU의 처리 단위 크기에 정렬되지 않는다. 이것도 문제다. 정리하면 TP는 대규모 클러스터에 적합하지 않고, 그래서 EP가 도입되었다. EP에 load balancing 문제가 있더라도, **먼저 아키텍처가 맞는지의 문제를 해결하고 그다음에 아키텍처 안의 어려움을 해결하는 것**이다. SGLang이 첫 번째 MLP layer에 적용한 DP Dense FFN 최적화에 대해서는, 이 최적화 때문에 코드 구현이 지나치게 복잡해져 이해하기 어렵다고 느낀다. Dense FFN 한 layer의 최적화가 주는 실제 이득도 그리 크지 않을 수 있어서 이 최적화는 논의 범위에서 제외한다. DeepSeek V3의 기술 보고서에도 Dense FFN에는 병렬을 사용하지 않았다고 쓰여 있다.

> In particular, we use 1-way Tensor Parallelism for the dense MLPs in shallow layers to save TP communication.

### PD 분리의 병렬

앞서 SGLang의 PD 분리를 소개할 때 DP와 TP 아래에서 Prefill과 Decode가 어떻게 상호작용하는지는 설명하지 않았으므로, 여기서 다룬다.

먼저 알아 둘 것은, 원리상 각 DP group이 유지하는 KV cache는 서로 일치하지는 않지만 완전하다는 점이다. Decode의 DP group은 Prefill의 임의의 DP group에서 KV cache를 가져올 수 있다. 물론 Decode의 DP group 안의 TP와 Prefill의 DP group 안의 TP가 같아야 한다. 그리고 하나의 DP group 안에서는 qkv_proj가 분할되어 있으므로 같은 token이라도 저장된 KV cache 값이 다르다. 따라서 TP rank끼리 일대일로 대응시켜 전송할 수밖에 없다. 이제 SGLang의 구체적인 구현을 보자.

SGLang은 bootstrap server에서 DP와 TP 정보를 유지하며, 2차원 테이블로 Prefill의 각 DP group 아래 각 TP rank의 ip와 port를 기록한다. port는 무작위로 생성된다.

``` python3
self.prefill_port_table[dp_group][tp_rank_in_dp_group] = {
    "rank_ip": rank_ip,
    "rank_port": rank_port,
}
```

각 req는 min_ib.py에서 요청을 생성할 때 세 개의 parameter를 함께 갖는다. bootstrap_host와 bootstrap_port(Prefill의 정보), 그리고 무작위로 생성된 bootstrap_room이며, 동시에 Prefill 인스턴스 하나와 Decode 인스턴스 하나로 보내진다. Decode 조회 쪽에서는 KVManger가 두 개의 테이블을 유지한다. 하나는 전역 prefill_dp_size_table로 각 Prefill 인스턴스의 DP size를 기록하고, 다른 하나는 connection_pool로 Decode 인스턴스의 각 TP rank마다 연결 정보, 즉 위에서 말한 Prefill의 각 DP group의 각 TP rank 주소를 유지한다.

``` python3
# Receiver 안의 구현
# Decode는 Prefill의 DP group 중 하나를 무작위로 고른다
self.target_dp_group = bootstrap_room % self.prefill_dp_size
# engine rank가 곧 TP rank이며, TP=32이면 0...31이다
bootstrap_key = f"{self.bootstrap_addr}_{self.kv_mgr.kv_args.engine_rank}"
if bootstrap_key not in self.kv_mgr.connection_pool:
    # Prefill의 어떤 DP group에서 Decode의 TP rank에 따라 Prefill의 대응 TP rank 주소를 얻는다
    self.bootstrap_info = self._get_bootstrap_info_from_server(
        self.kv_mgr.kv_args.engine_rank,
        self.target_dp_group,
    )
    if self.bootstrap_info is None:
        logger.error(
            f"Could not fetch bootstrap info for engine rank: {self.kv_mgr.kv_args.engine_rank}"
        )
    else:
        self.kv_mgr.connection_pool[bootstrap_key] = self.bootstrap_info
        # Register kv_args only once to prefill KVManager according to the info fetched from the bootstrap server
        # 그다음 Decode 쪽 어떤 TP rank의 KV cache 물리 주소를 Prefill의 대응 TP rank로 보낸다
        self._register_kv_args()
else:
    self.bootstrap_info = self.kv_mgr.connection_pool[bootstrap_key]
```

DeepSeek의 설정을 예로 들어 보자. TP=32, DP=8이고 node마다 카드가 8장이면, node 하나에 DP group이 2개 있고 DP group 하나에 TP rank가 4개 있다. engine_rank=9라면 두 번째 카드 위의 DP group 2의 TP rank 1이다. target_dp_group으로 DP group 0을 무작위로 골랐다면, 얻게 되는 것은 Prefill의 0번째 DP group에서 TP rank가 1인 주소다. 즉 DP group은 달라도 되지만 TP rank는 반드시 같아야 한다.

위의 설정은 Attention layer에서는 ATP=4가 되며, DP group 안에서 allreduce를 한 번 해야 한다. 만약 DP=32라면 Attention에는 통신이 없다. MoE layer에서는 EP와 TP가 상호 배타적이므로 있는 카드를 전부 쓰게 되어 EP=32다.
