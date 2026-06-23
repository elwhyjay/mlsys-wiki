# LightX2V Ulysses Attention 구현 학습 노트

이 글은 LightX2V 안의 Ulysses Attention 구현 방식과 tensor shape 추론을 기록한다. 다음 file을 다룬다.

- `LightX2V/lightx2v/common/ops/attn/ulysses_attn.py`
- `LightX2V/lightx2v/common/ops/attn/utils/all2all.py`(`all2all_head2seq` / `all2all_seq2head`)

performance 관련 내용은 implementation mechanism 수준의 영향 요인만 이야기한다. 직접 측정하지 않았으므로 확정적인 설명은 하지 않는다. 이 글을 쓴 이유는 Ulysses Attention 구현을 조금 잊어버려서 주말에 다시 직접 따라가 보기 위해서다. code repository는 https://github.com/ModelTC/LightX2V 이다.

> 필자 주: 아래 code snippet에는 내가 주석을 추가하지 않았다. 안의 주석은 LightX2V에서 code snippet을 복사할 때 이미 붙어 있던 것이다. line number는 commit `bef76dc298983053df2506d8eaaa97d1895ec077` 기준으로 맞췄다.

## 0x0. 배경

multimodal/video scenario에서 image 또는 video token sequence는 보통 text token sequence보다 훨씬 길다. single-card에서 FlashAttention은 주로 GPU memory 사용량과 memory access behavior를 최적화한다. multi-card parallelism에서는 추가로 다음을 처리해야 한다.

- 긴 sequence에서 parallel partition 방식(sequence split, head split, 또는 둘의 조합)
- partition 방식에서 생기는 communication(collective)과 data reorder overhead

LightX2V의 Ulysses 구현은 하나의 layout transform으로 요약할 수 있다.

- input side: image token은 **sequence shard + full heads** layout 사용
- Attention side: all-to-all 한 번으로 **full sequence + head shard**로 변환해 각 rank가 담당하는 head subset을 계산
- output side: 다시 all-to-all 한 번으로 image output을 **sequence shard + full heads**로 복원

text 부분은 이 구현에서 head 기준 aggregation(`all_gather`)으로 full heads를 다시 만든다.

## 0x1. symbol과 shape convention

다음처럼 표기한다. code symbol을 그대로 사용한다.

- `P`: `world_size = dist.get_world_size(seq_p_group)`(parallel group size)
- `H`: attention head 수(code variable `heads`)
- `D`: 각 head의 dimension(code variable `hidden_dims`)
- `shardH = H // P`: rank마다 담당하는 head 수(`H % P == 0` 필요)

sequence length는 image와 text를 구분한다. 이는 `ulysses_attn.py`의 segment logic에 대응한다.

- `imgShardS`: 현재 rank의 image token length(`img_qkv_len`, sharded 이후)
- `imgS = imgShardS * P`: global image token length(`global_img_seqlen`)
- `txtS`: text token length(`txt_qkv_len`)
- `S = imgS + txtS`: concat 이후 total length. 여기서는 `txt_mask_len` branch는 우선 무시한다.

FlashAttention wrapper의 output shape convention은 `LightX2V/lightx2v/common/ops/attn/flash_attn.py`를 보면 다음과 같다.

- input `q/k/v`: `[T, shardH, D]` 또는 4D에서 3D로 reshape
- output: `[T, shardH * D]`(flatten output)

따라서 `ulysses_attn.py`에서 attention output을 다시 3D shape로 복원하는 모든 step은 `D`와 `shardH`의 일관성에 의존한다.

## 0x2. `UlyssesAttnWeight`(default implementation) 분석

target file: `LightX2V/lightx2v/common/ops/attn/ulysses_attn.py`, class: `UlyssesAttnWeight`.

아래에서는 `enable_head_parallel=False`의 main path를 중심으로 설명한다.

### 0x2.1 input shape와 image/text split

function input comment는 다음과 같이 적혀 있다.

- `q/k/v`: `[shard_seqlen, heads, hidden_dims]`

구현에서는 4D input도 지원하며, 들어오면 3D로 reshape한다.

- 4D: `[B, S, H, D]`
- 3D: `[T, H, D]`, 여기서 `T = B * S`

대응 code(4D -> 3D reshape):

```47:50:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        if len(q.shape) == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
```

`img_first=True`일 때 image와 text length 기준으로 split한다.

- `img_q/img_k/img_v`: `[imgShardS, H, D]`
- `txt_q/txt_k/txt_v`: `[txtS, H, D]`

대응 code(img/text segmentation과 `.contiguous()`):

```56:96:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        # Get sequence length and text-related lengths
        if img_first:
            img_qkv_len = slice_qkv_len
            if len(cu_seqlens_qkv) == 3:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len  # length of text query/key/value
                txt_mask_len = cu_seqlens_qkv[2] - slice_qkv_len  # text mask length
            elif len(cu_seqlens_qkv) == 2:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len  # length of text query/key/value
                txt_mask_len = None
        else:
            # assert len(cu_seqlens_qkv) == 2
            txt_qkv_len = slice_qkv_len
            img_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len
            txt_mask_len = None
 
        # Split image and text query/key/value
        if img_first:
            img_q, img_k, img_v = q[:img_qkv_len, :, :].contiguous(), k[:img_qkv_len, :, :].contiguous(), v[:img_qkv_len, :, :].contiguous()
            txt_q, txt_k, txt_v = q[img_qkv_len:, :, :].contiguous(), k[img_qkv_len:, :, :].contiguous(), v[img_qkv_len:, :, :].contiguous()
        else:
            txt_q, txt_k, txt_v = q[:txt_qkv_len, :, :].contiguous(), k[:txt_qkv_len, :, :].contiguous(), v[:txt_qkv_len, :, :].contiguous()
            img_q, img_k, img_v = q[txt_qkv_len:, :, :].contiguous(), k[txt_qkv_len:, :, :].contiguous(), v[txt_qkv_len:, :, :].contiguous()
```

### 0x2.2 image QKV layout transform(seq shard -> head shard)

image QKV를 stack한다.

- `img_qkv = stack([img_q, img_k, img_v], dim=0)`: `[3, imgShardS, H, D]`

head dimension `H`를 `[P, shardH]`로 split한다.

- `reshape(3, imgShardS, P, shardH, D)`: `[3, imgShardS, P, shardH, D]`

all-to-all 준비를 위해 binning dimension을 정리한다. `P`를 dim0으로 둔다.

- `permute(2, 1, 0, 3, 4)`: `[P, imgShardS, 3, shardH, D]`

all-to-all 실행:

- `dist.all_to_all_single(output_qkv, input_t, group=seq_p_group)`
- `output_qkv` shape는 여전히 `[P, imgShardS, 3, shardH, D]`

그 뒤 standard Q/K/V로 정리한다.

- `output_qkv.reshape(imgS, 3, shardH, D).transpose(0, 1)`: `[3, imgS, shardH, D]`
- `shard_img_q/k/v`: `[imgS, shardH, D]`

대응 code(stack + reshape, 그리고 두 branch의 permute/all_to_all/reshape):

```97:177:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        img_qkv = torch.stack([img_q, img_k, img_v], dim=0).reshape(3, img_qkv_len, world_size, shard_heads, hidden_dims)
        original_dtype = img_qkv.dtype
 
        if enable_head_parallel:
            img_qkv = img_qkv.permute(3, 2, 1, 0, 4).contiguous()  # (shard_heads, world_size, img_qkv_len, 3, hidden_dims)
            output_qkv = torch.empty_like(img_qkv)
            # ... per-head all_to_all_single + reshape ...
            qkv = output_qkv[h].reshape(global_img_seqlen, 3, single_head, hidden_dims).transpose(0, 1)
            shard_img_q = qkv[0]  # (global_img_seqlen, single_head, hidden_dims)
            shard_img_k = qkv[1]
            shard_img_v = qkv[2]
        else:
            img_qkv = img_qkv.permute(2, 1, 0, 3, 4).contiguous()  # (world_size, img_qkv_len, 3, shard_heads, hidden_dims)
            # ... all_to_all_single ...
            qkv = output_qkv.reshape(global_img_seqlen, 3, shard_heads, hidden_dims).transpose(0, 1)
            shard_img_q = qkv[0]  # (global_img_seqlen, shard_head, hidden_dims)
            shard_img_k = qkv[1]
            shard_img_v = qkv[2]
```

대응 code(`enable_head_parallel=True`: per-head all_to_all + reshape to `[global_img_seqlen, 3, 1, hidden_dims]`):

```100:156:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        if enable_head_parallel:
            img_qkv = img_qkv.permute(3, 2, 1, 0, 4).contiguous()  # (shard_heads, world_size, img_qkv_len, 3, hidden_dims)
            output_qkv = torch.empty_like(img_qkv)
 
            # Communicate image query/key/value
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(shard_heads, world_size, img_qkv_len, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(shard_heads, world_size, img_qkv_len, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                comm_fp8_works = []
                comm_scale_works = []
                for h in range(shard_heads):
                    work_fp8 = dist.all_to_all_single(output_qkv_fp8[h], img_qkv_fp8[h], group=seq_p_group, async_op=True)
                    work_scale = dist.all_to_all_single(output_qkv_scale[h], img_qkv_scale[h], group=seq_p_group, async_op=True)
                    comm_fp8_works.append(work_fp8)
                    comm_scale_works.append(work_scale)
            else:
                comm_works = []
                for h in range(shard_heads):
                    work = dist.all_to_all_single(output_qkv[h], img_qkv[h], group=seq_p_group, async_op=True)
                    comm_works.append(work)
 
            # Compute attention one head at a time
            single_head = 1
            head_attns = []
            for h in range(shard_heads):
                if use_fp8_comm:
                    comm_fp8_works[h].wait()
                    comm_scale_works[h].wait()
                    output_qkv[h] = dequant_fp8_vllm(output_qkv_fp8[h], output_qkv_scale[h], original_dtype)
                else:
                    comm_works[h].wait()
 
                qkv = output_qkv[h].reshape(global_img_seqlen, 3, single_head, hidden_dims).transpose(0, 1)
                shard_img_q = qkv[0]  # (global_img_seqlen, single_head, hidden_dims)
                shard_img_k = qkv[1]
                shard_img_v = qkv[2]
 
                # Process text query/key/value and select the current head of the current process
                shard_txt_q = txt_q[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_k = txt_k[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_v = txt_v[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
 
                # Merge image and text query/key/value
                q = torch.cat((shard_img_q, shard_txt_q), dim=0)
                k = torch.cat((shard_img_k, shard_txt_k), dim=0)
                v = torch.cat((shard_img_v, shard_txt_v), dim=0)
 
                # Call attention function to compute attention result
                head_attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)
                head_attns.append(head_attn)
 
            # Merge all local head attention results for the current process
            attn = torch.cat(head_attns, dim=1)
```

대응 code(`enable_head_parallel=False`: one all_to_all + reshape to `[global_img_seqlen, 3, shard_heads, hidden_dims]`):

```157:192:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        else:
            img_qkv = img_qkv.permute(2, 1, 0, 3, 4).contiguous()  # (world_size, img_qkv_len, 3, shard_heads, hidden_dims)
 
            # Communicate image query/key/value
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(world_size, img_qkv_len, shard_heads, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(world_size, img_qkv_len, shard_heads, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                dist.all_to_all_single(output_qkv_fp8, img_qkv_fp8, group=seq_p_group)
                dist.all_to_all_single(output_qkv_scale, img_qkv_scale, group=seq_p_group)
                output_qkv = dequant_fp8_vllm(output_qkv_fp8, output_qkv_scale, original_dtype)
            else:
                output_qkv = torch.empty_like(img_qkv)
                dist.all_to_all_single(output_qkv, img_qkv, group=seq_p_group)
 
            # Finish attention computation
            qkv = output_qkv.reshape(global_img_seqlen, 3, shard_heads, hidden_dims).transpose(0, 1)
            shard_img_q = qkv[0]  # (global_img_seqlen, shard_head, hidden_dims)
            shard_img_k = qkv[1]
            shard_img_v = qkv[2]
 
            # Process text query/key/value and select current heads for the current process
            shard_txt_q = txt_q[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_k = txt_k[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_v = txt_v[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
 
            # Merge image and text query/key/value
            q = torch.cat((shard_img_q, shard_txt_q), dim=0)
            k = torch.cat((shard_img_k, shard_txt_k), dim=0)
            v = torch.cat((shard_img_v, shard_txt_v), dim=0)
 
            # Call attention function to compute attention result
            attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)
```

이 step의 결과는 각 rank가 full image sequence(length `imgS`)와 자신이 담당하는 head subset(`shardH`)을 갖는다는 것이다.

### 0x2.3 text branch: head slice

text QKV는 seq dimension all-to-all을 하지 않고, head dimension 기준으로 현재 rank의 head subset만 선택한다.

- `shard_txt_q/k/v`: `[txtS, shardH, D]`

대응 code(두 branch 모두 text에 head slice 적용):

```140:183:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
                # Process text query/key/value and select the current head of the current process
                shard_txt_q = txt_q[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_k = txt_k[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
                shard_txt_v = txt_v[:, (cur_rank * shard_heads + h) : (cur_rank * shard_heads + h + 1), :]
...
            # Process text query/key/value and select current heads for the current process
            shard_txt_q = txt_q[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_k = txt_k[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
            shard_txt_v = txt_v[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
```

### 0x2.4 concat and compute attention(output flatten 주의)

image와 text를 concat한다.

- `q/k/v = cat([shard_img_*, shard_txt_*], dim=0)`: `[imgS + txtS, shardH, D]`

`attention_module.apply(...)`, 예를 들어 `FlashAttn2Weight/FlashAttn3Weight`를 호출하면 다음을 얻는다.

- `attn`: `[imgS + txtS, shardH * D]`

대응 code(q/k/v concat 후 attention_module.apply 호출):

```145:191:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
                # Merge image and text query/key/value
                q = torch.cat((shard_img_q, shard_txt_q), dim=0)
                k = torch.cat((shard_img_k, shard_txt_k), dim=0)
                v = torch.cat((shard_img_v, shard_txt_v), dim=0)
 
                # Call attention function to compute attention result
                head_attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)
...
            # Merge image and text query/key/value
            q = torch.cat((shard_img_q, shard_txt_q), dim=0)
            k = torch.cat((shard_img_k, shard_txt_k), dim=0)
            v = torch.cat((shard_img_v, shard_txt_v), dim=0)
 
            # Call attention function to compute attention result
            attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, **kwargs)
```

구현에서는 varlen interface에 맞게 `cu_seqlens_qkv`와 `max_seqlen_qkv`를 다시 만든다. 이 부분은 tensor shape 추론 결론을 바꾸지 않는다.

### 0x2.5 output split과 layout 복원(head shard -> seq shard)

sequence 기준으로 output을 split한다(`img_first=True`).

- `img_attn`: `[imgS, shardH*D]`
- `txt_attn`: `[txtS, shardH*D]`

대응 code(attn에서 img/text output 분리):

```193:197:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        # Split image and text attention results
        if img_first:
            img_attn, txt_attn = attn[:global_img_seqlen, :], attn[global_img_seqlen:]
        else:
            txt_attn, img_attn = attn[:txt_qkv_len, :], attn[txt_qkv_len:]
```

#### 0x2.5.1 text: `all_gather`로 full heads 복원

`txt_attn`에 대해 `all_gather`를 수행하고 dim=1로 concat한다.

- each rank: `[txtS, shardH*D]`
- concat after gather: `[txtS, P*shardH*D] = [txtS, H*D]`

대응 code(all_gather + cat):

```202:206:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        # Gather text attention results from all processes
        gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(world_size)]
        dist.all_gather(gathered_txt_attn, txt_attn, group=seq_p_group)
        txt_attn = torch.cat(gathered_txt_attn, dim=1)  # Merge text attention results from all processes
```

#### 0x2.5.2 image: `_reshape_img_attn()` + `all2all_head2seq()`

먼저 flatten output을 3D로 복원한다.

- `img_attn.reshape(imgS, shardH, D)`: `[imgS, shardH, D]`

`all2all_head2seq` 호출:

- input: `[seq_len, heads/P, D]`, 여기서는 `[imgS, shardH, D]`
- output: `[seq_len/P, heads, D]`, 여기서는 `[imgShardS, H, D]`

마지막으로 flatten한다.

- `[imgShardS, H*D]`

이제 image output은 **sequence shard + full heads** layout으로 돌아온다.

대응 code(img_attn reshape -> all2all_head2seq -> reshape flatten):

```215:231:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
    @torch.compiler.disable
    def _reshape_img_attn(self, img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group, use_fp8_comm):
        img_attn = img_attn.reshape(world_size * shard_seqlen, shard_heads, hidden_dims)  # Reshape image attention result
 
        # Convert head format back to sequence format
        if use_fp8_comm:
            original_dtype = img_attn.dtype
            original_shape = img_attn.shape
            img_attn_fp8, attn_scale = quant_fp8_vllm(img_attn.reshape(-1, original_shape[-1]))
            img_attn_fp8 = all2all_head2seq(img_attn_fp8.reshape(original_shape), group=seq_p_group)
            attn_scale = all2all_head2seq(attn_scale.reshape(original_shape[0], original_shape[1], 1), group=seq_p_group)
            img_attn = dequant_fp8_vllm(img_attn_fp8, attn_scale, original_dtype)
        else:
            img_attn = all2all_head2seq(img_attn, group=seq_p_group)
 
        img_attn = img_attn.reshape(shard_seqlen, -1)  # Reshape to [shard_seqlen, -1]
        return img_attn
```

대응 code(final output attn concat, `img_first`에 따라 concat order 결정):

```207:213:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
        # Merge image and text attention results
        if img_first:
            attn = torch.cat([img_attn, txt_attn], dim=0)
        else:
            attn = torch.cat([txt_attn, img_attn], dim=0)
 
        return attn  # Return final attention result
```

## 0x3. `all2all_head2seq` dimension derivation

`all2all_head2seq`의 목표는 input `X: [S, H//P, D]`를 output `[S//P, H, D]`로 바꾸는 것이다. 여기에는 `S % P == 0`, `H % P == 0`이 필요하다. 즉 먼저 `S`를 `P`개로 나누고 `[P, shardS, shardH, D]`로 reshape한다. 여기서 `shardS = S//P`, `shardH = H//P`다. 이어 transpose로 `[P, shardH, shardS, D]`를 만들고, `dist.all_to_all_single`을 실행하면 shape는 유지된다. 마지막으로 앞 두 dimension을 merge해 `[H, shardS, D]`를 얻고 다시 transpose해 `[shardS, H, D]`가 된다. `all2all_seq2head`는 반대 방향 mapping을 수행한다.

## 0x4. `enable_head_parallel` branch의 구조적 차이

`enable_head_parallel=True`일 때 default implementation은 `shardH`개의 head를 loop로 나누어 처리한다.

- 각 local head에 대해 all-to-all을 한 번씩 수행하고 single-head QKV로 attention 계산
- 최종적으로 각 head의 output을 head dimension에서 concat

이 branch의 주요 차이는 collective call 수와 attention kernel call 수가 크게 늘어난다는 점이다. 실제 효과는 communication과 compute가 충분히 overlap될 수 있는지, 그리고 구체적인 runtime environment의 scheduling과 latency 특성에 달려 있다.

## 0x5. `use_fp8_comm` communication quantization branch

구현은 `use_fp8_comm=True` option을 제공하며, image QKV의 all-to-all과 image output return path에 다음을 도입한다.

- FP8 quantization(`quant_fp8_vllm`)과 scale
- FP8과 scale의 communication
- dequantization(`dequant_fp8_vllm`)

가능한 영향 요인:

- communication bytes: FP8 data는 보통 bf16/fp16보다 작지만 추가 scale transfer가 포함된다.
- compute overhead: quantization과 dequantization operation이 추가된다.
- availability: `vllm._custom_ops` 구현과 runtime environment에 의존한다.
- numerical error: FP8 quantization이 유발하며, 허용 가능성은 model과 task에 따라 다르다.

대응 code(default implementation: FP8 quantization/scale reshape와 dequant, 두 branch에 각각 위치):

```105:117:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(shard_heads, world_size, img_qkv_len, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(shard_heads, world_size, img_qkv_len, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                comm_fp8_works = []
                comm_scale_works = []
                for h in range(shard_heads):
                    work_fp8 = dist.all_to_all_single(output_qkv_fp8[h], img_qkv_fp8[h], group=seq_p_group, async_op=True)
                    work_scale = dist.all_to_all_single(output_qkv_scale[h], img_qkv_scale[h], group=seq_p_group, async_op=True)
                    comm_fp8_works.append(work_fp8)
                    comm_scale_works.append(work_scale)
```

```161:169:LightX2V/lightx2v/common/ops/attn/ulysses_attn.py
            if use_fp8_comm:
                img_qkv_fp8, img_qkv_scale = quant_fp8_vllm(img_qkv.reshape(-1, hidden_dims))
                img_qkv_fp8 = img_qkv_fp8.reshape(world_size, img_qkv_len, shard_heads, 3, hidden_dims)
                img_qkv_scale = img_qkv_scale.reshape(world_size, img_qkv_len, shard_heads, 3, 1)
                output_qkv_fp8 = torch.empty_like(img_qkv_fp8)
                output_qkv_scale = torch.empty_like(img_qkv_scale)
                dist.all_to_all_single(output_qkv_fp8, img_qkv_fp8, group=seq_p_group)
                dist.all_to_all_single(output_qkv_scale, img_qkv_scale, group=seq_p_group)
                output_qkv = dequant_fp8_vllm(output_qkv_fp8, output_qkv_scale, original_dtype)
```

## 0x6. 정리

LightX2V의 Ulysses Attention 구현은 image all-to-all 두 번(input side와 output side)으로 layout transform을 수행해 attention computation이 **full sequence + head shard** layout 위에서 일어나게 한다. text 부분은 `all_gather`로 full heads를 다시 만든다. 구현은 동시에 다음 option도 제공한다.

- `enable_head_parallel`: 더 fine-grained한 head-level 처리
- `use_fp8_comm`: FP8 communication quantization

이 option들의 효과는 runtime environment, communication backend, data scale의 영향을 받으므로 실제 configuration과 함께 평가해야 한다. 개인적으로 보기에는 video model용으로 들어간 구현처럼 보인다. 여기서는 Ulysses Attention의 shape derivation과 flow만 기록하고, 더 많은 추측은 하지 않는다.

