## 0x0. 머리말

최근에서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SGLang Diffusion)의관련 내용구현，관련 내용이다 SGLang 관련 내용의관련 내용모델관련 내용지원 Wan、Hunyuan、Qwen-Image、Flux 관련 내용의관련 내용와관련 내용생성한다모델。필자는관련 내용로 FLUX.1-dev 로관련 내용기록관련 내용하필자는대해관련 내용구현의관련 내용주요관련 내용모델관련 내용그리고row관련 내용와 attention backend 관련 내용개관련 내용

## 0x1. 관련 내용

SGLang Diffusion 이다기반으로 SGLang 의 serving 관련 내용와서구현의，이 부분은 원문의 해당 기술 설명을 이어서 서술한다이다 `ComposedPipelineBase` 와 `PipelineStage` 의관련 내용모드（코드에서 `python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py` 와 `stages/base.py`）。각개 stage 관련 내용개관련 내용의관련 내용가능，이 부분은 원문의 해당 기술 설명을 이어서 서술한다가서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (VAE)통해관련 내용이들 stage 관련 내용가능로관련 내용완전한의관련 내용

관련 내용개관련 내용의 pipeline 된다관련 내용이들 stage：InputValidationStage（입력검증）、TextEncodingStage（이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (ConditioningStage TimestepPreparationStage LatentPreparationStage latent DenoisingStage)가서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (DecodingStage VAE block)의관련 내용추가새모델또는관련 내용있다 pipeline 모두이 부분은 원문의 해당 기술 설명을 이어서 서술한다

## 0x2. FLUX.1-dev 모델구현상세 해설

관련 내용하와서필자는로 FLUX.1-dev 로관련 내용보다보다 SGLang Diffusion 이다관련 내용와관련 내용 (row)모델의。

### 2.1 Pipeline 설정

FLUX.1-dev 의 pipeline 설정관련 내용에서 `FluxPipelineConfig` 중（`configs/pipeline_configs/flux.py`）：

```python
@dataclass
class FluxPipelineConfig(ImagePipelineConfig):
    """Configuration for the FLUX pipeline."""
    
    embedded_cfg_scale: float = 3.5
    task_type: ModelTaskType = ModelTaskType.T2I
    
    # DiT 설정
    dit_config: DiTConfig = field(default_factory=FluxConfig)
    
    # VAE 설정
    vae_config: VAEConfig = field(default_factory=FluxVAEConfig)
    
    # Text encoder 설정（CLIP + T5）
    text_encoder_configs: tuple[EncoderConfig,...] = field(
        default_factory=lambda: (CLIPTextConfig(), T5Config())
    )
```

이설정이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (FLUX.1-dev)의관련 내용있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (DiT)모델）、VAE（이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Text Encoders CLIP)와 T5 관련 내용개이 부분은 원문의 해당 기술 설명을 이어서 서술한다

### 2.2 모델관련 내용

FLUX.1-dev 의 transformer 관련 내용에서 `FluxTransformer2DModel` 중구현（`runtime/models/dits/flux.py`）：

```python
class FluxTransformer2DModel(CachableDiT):
    def __init__(self, config: FluxConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        
        # 관련 내용
        self.rotary_emb = FluxPosEmbed(theta=10000, axes_dim=self.config.axes_dims_rope)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(...)
        self.context_embedder = ReplicatedLinear(...)
        self.x_embedder = ReplicatedLinear(...)
        
        # Transformer blocks（관련 내용
        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(...) for _ in range(self.config.num_layers)
        ])
        
        # Single transformer blocks
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(...) for _ in range(self.config.num_single_layers)
        ])
```

FLUX 관련 내용사용관련 내용의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (transformer_blocks)와관련 내용의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (attention 19layer single_transformer_blocks)만관련 내용의 attention（38layer）。이관련 내용이다관련 내용있다관련 내용의。

### 2.3 Pipeline Stages 상세 해설

FLUX.1-dev 의 pipeline 에 의해로하 stages 관련 내용완전한코드에서 `runtime/pipelines/flux.py` 의 `create_pipeline_stages` 관련 내용

```python
def create_pipeline_stages(self, server_args: ServerArgs):
    # 1. 입력검증
    self.add_stage(
        stage_name="input_validation_stage", 
        stage=InputValidationStage()
    )
    
    # 2. 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CLIP + T5)
    self.add_stage(
        stage_name="prompt_encoding_stage_primary",
        stage=TextEncodingStage(
            text_encoders=[
                self.get_module("text_encoder"),      # CLIP
                self.get_module("text_encoder_2"),    # T5
            ],
            tokenizers=[
                self.get_module("tokenizer"),
                self.get_module("tokenizer_2"),
            ],
        ),
    )
    
    # 3. 관련 내용
    self.add_stage(
        stage_name="conditioning_stage", 
        stage=ConditioningStage()
    )
    
    # 4. 이 부분은 원문의 해당 기술 설명을 이어서 서술한다
    self.add_stage(
        stage_name="timestep_preparation_stage",
        stage=TimestepPreparationStage(
            scheduler=self.get_module("scheduler"),
            prepare_extra_set_timesteps_kwargs=[prepare_mu],
        ),
    )
    
    # 5. Latent 관련 내용
    self.add_stage(
        stage_name="latent_preparation_stage",
        stage=LatentPreparationStage(
            scheduler=self.get_module("scheduler"),
            transformer=self.get_module("transformer"),
        ),
    )
    
    # 6. 가서관련 내용
    self.add_stage(
        stage_name="denoising_stage",
        stage=DenoisingStage(
            transformer=self.get_module("transformer"),
            scheduler=self.get_module("scheduler"),
        ),
    )
    
    # 7. VAE 관련 내용
    self.add_stage(
        stage_name="decoding_stage", 
        stage=DecodingStage(vae=self.get_module("vae"))
    )
```

`TextEncodingStage`（`pipelines_core/stages/text_encoding.py`）담당할 것이다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (prompt embedding)

```python
class TextEncodingStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 관련 내용사용 CLIP 와 T5 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (prompt)
        prompt_embeds_list, prompt_masks_list, pooler_embeds_list = self.encode_text(
            prompt_text,
            server_args,
            encoder_index=all_indices,
            return_attention_mask=True,
        )
        
        # 만약관련 내용사용 CFG，도이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (negative prompt)
        if batch.do_classifier_free_guidance:
            neg_embeds_list, neg_masks_list, neg_pooler_embeds_list = self.encode_text(
                batch.negative_prompt,
                server_args,
                encoder_index=all_indices,
                return_attention_mask=True,
            )
```

FLUX 관련 내용사용관련 내용개이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CLIP pooled embeddings)사용된다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (T5 column embeddings)사용된다이 부분은 원문의 해당 기술 설명을 이어서 서술한다

`DenoisingStage`（`pipelines_core/stages/denoising.py`）이다관련 내용의 stage，실행한다관련 내용가서관련 내용

```python
class DenoisingStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 초기화 latents
        latents = batch.latents
        
        # 관련 내용가서관련 내용
        for i, t in enumerate(timesteps):
            # 관련 내용입력
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            # Transformer 전관련 내용
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                timestep=t,
                freqs_cis=freqs_cis,
            )
            
            # 갱신 latents
            latents = self.scheduler.step(noise_pred, t, latents)
```

### 2.4 모델로드관련 내용

SGLang Diffusion 관련 내용사용 `PipelineComponentLoader` 와서로드관련 내용개관련 내용`runtime/loader/component_loader.py`），관련 내용의로드관련 내용에서 `ComposedPipelineBase` 의 `load_modules` 관련 내용중：

```python
def load_modules(self, server_args: ServerArgs) -> dict[str, Any]:
    model_index = self._load_config()  # 읽기 model_index.json
    
    components = {}
    for module_name, (transformers_or_diffusers, architecture) in model_index.items():
        if module_name not in required_modules:
            continue
            
        # 로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (block)
        module = PipelineComponentLoader.load_module(
            module_name=module_name,
            component_model_path=component_model_path,
            transformers_or_diffusers=transformers_or_diffusers,
            server_args=server_args,
        )
        components[module_name] = module
    
    return components
```

로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다읽기 `model_index.json` 얻는다이 부분은 원문의 해당 기술 설명을 이어서 서술한다설정로드각개이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (transformer), vae, text_encoder 관련 내용응용 TP/SP 관련 내용그리고row관련 내용마지막으로반환한다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (pipeline)사용。

### 2.5 모델weight로드관련 내용

SGLang Diffusion 의weight로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다대상으로아니관련 내용의관련 내용사용아니관련 내용의로드관련 내용필자는관련 내용하이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (block)의구현。

**로드관련 내용모드**

SGLang 로각이 부분은 원문의 해당 기술 설명을 이어서 서술한다모두구현관련 내용의 Loader（`runtime/loader/component_loader.py`）：

```python
class ComponentLoader:
    def load(self, component_model_path, server_args, module_name, transformers_or_diffusers):
        # 관련 내용로드관련 내용버전
        try:
            component = self.load_customized(component_model_path, server_args, module_name)
            source = "customized"
        except Exception:
            # 관련 내용까지관련 내용버전（transformers/diffusers）
            component = self.load_native(component_model_path, server_args, transformers_or_diffusers)
            source = "native"
        return component
```

관련 내용의좋은관련 내용이다관련 내용사용 SGLang 최적화관련 내용의구현，만약로드관련 내용다시관련 내용까지관련 내용의 transformers/diffusers 구현，이 부분은 원문의 해당 기술 설명을 이어서 서술한다

**Transformer weight로드（FSDP 관련 내용**

대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Transformer DiT)큰모델，SGLang 관련 내용사용 FSDP（Fully Sharded Data Parallel）와서로드weight：

```python
class TransformerLoader(ComponentLoader):
    def load_customized(self, component_model_path, server_args, *args):
        # 1. 읽기설정
        config = get_diffusers_component_config(model_path=component_model_path)
        dit_config = server_args.pipeline_config.dit_config
        dit_config.update_model_arch(config)
        
        # 2. 관련 내용까지관련 내용있다 safetensors 파일
        safetensors_list = _list_safetensors_files(component_model_path)
        
        # 3. 관련 내용사용 FSDP 로드모델
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={"config": dit_config, "hf_config": hf_config},
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=server_args.dit_cpu_offload,
            default_dtype=torch.bfloat16,
        )
        return model.eval()
```

FSDP 의관련 내용이다가능로할 것이다모델파라미터관련 내용까지많은개 GPU 상，지원 CPU offload，관련 내용이다 23.8GB 의 FLUX transformer 도가능에서관련 내용있다관련 내용의 GPU 상로드。

**Text Encoder weight로드（관련 내용로드）**

대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Text Encoder SGLang)사용더관련 내용의관련 내용로드관련 내용

```python
class TextEncoderLoader(ComponentLoader):
    def load_model(self, model_path, model_config, server_args, dtype="fp16"):
        # 1. 관련 내용초기화생성한다관련 내용모델
        with skip_init_modules():
            model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
            model = model_cls(model_config)
        
        # 2. 관련 내용로드weight
        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(
            self._get_all_weights(model, model_path, to_cpu=should_offload)
        )
        
        # 3. 관련 내용까지관련 내용
        model = model.to(local_torch_device)
        
        # 4. 만약이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CPU offload)사용 FSDP
        if should_offload:
            shard_model(
                model,
                cpu_offload=True,
                reshard_after_forward=True,
                mesh=mesh["offload"],
            )
        return model.eval()
```

여기의핵심이다 `skip_init_modules` 상하이 부분은 원문의 해당 기술 설명을 이어서 서술한다된다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (PyTorch)의기본파라미터초기화，이 부분은 원문의 해당 기술 설명을 이어서 서술한다와관련 내용그다음통해 `_get_all_weights` 얻는다weight이 부분은 원문의 해당 기술 설명을 이어서 서술한다로드weight。

**weight관련 내용구현**

`_get_all_weights` 반환한다관련 내용개생성한다관련 내용개읽기 safetensors 파일중의weight：

```python
def _get_weights_iterator(self, source, to_cpu):
    hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
        source.model_or_path, source.fall_back_to_pt, source.allow_patterns_overrides
    )
    
    if use_safetensors:
        weights_iterator = safetensors_weights_iterator(hf_weights_files, to_cpu=to_cpu)
    else:
        weights_iterator = pt_weights_iterator(hf_weights_files, to_cpu=to_cpu)
    
    # 응용전관련 내용
    return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
```

관련 내용로드의좋은관련 내용이다아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다있다weight로드까지관련 내용가능로관련 내용로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다

**이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)로드관련 내용**

각개모델모두구현관련 내용의 `load_weights` 관련 내용와서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight CLIP)의구현：

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    # QKV 융합관련 내용
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]
    
    params_dict = dict(self.named_parameters())
    loaded_params = set()
    
    for name, loaded_weight in weights:
        # 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (q_proj), k_proj, v_proj -> qkv_proj 의관련 내용
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in name:
                model_param_name = name.replace(weight_name, param_name)
                if model_param_name in params_dict:
                    param = params_dict[model_param_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(model_param_name)
                break
        else:
            # 기본로드관련 내용
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
    
    return loaded_params
```

여기의 `weight_loader` 이다관련 내용개가능관련 내용의함수，가능로이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (QKV)융합、weight분할、이 부분은 원문의 해당 기술 설명을 이어서 서술한다

**VAE weight로드（관련 내용**

VAE 관련 내용대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용 `load_state_dict`：

```python
class VAELoader(ComponentLoader):
    def load_customized(self, component_model_path, server_args, *args):
        # 1. 생성한다모델
        vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
        vae = vae_cls(vae_config).to(target_device)
        
        # 2. 로드weight
        safetensors_list = _list_safetensors_files(component_model_path)
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(loaded, strict=False)
        
        return vae.eval()
```

VAE 관련 내용작은（168 MiB），그래서가능로이 부분은 원문의 해당 기술 설명을 이어서 서술한다로드관련 내용있다weight。

**정리관련 내용하weight로드의관련 내용**：

1. **이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer)** ：아니관련 내용사용아니관련 내용의 Loader，각개 Loader 있다관련 내용와관련 내용로드관련 내용
2. **관련 내용로드** ：대해관련 내용큰모델（Text Encoder、Transformer），관련 내용사용생성한다관련 내용로드，관련 내용
3. **FSDP 지원** ：큰모델지원 FSDP 관련 내용와 CPU offload，가능로에서있다관련 내용하로드관련 내용큰모델
4. **weight관련 내용** ：각개모델가능로관련 내용`load_weights` 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)와융합
5. **관련 내용초기화** ：관련 내용사용 `skip_init_modules` 이 부분은 원문의 해당 기술 설명을 이어서 서술한다초기화파라미터

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SGLang Diffusion)가능로높은관련 내용로드관련 내용의모델，이 부분은 원문의 해당 기술 설명을 이어서 서술한다좋은의관련 내용와관련 내용

## 0x3. 그리고row관련 내용상세 해설

관련 내용하와서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SGLang Diffusion)의그리고row이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (block)이다관련 내용높은성능의핵심。관련 내용지원많은관련 내용그리고row관련 내용필자는관련 내용개개와서보다。

### 3.1 Tensor Parallelism (TP)

TP 관련 내용이다모델의파라미터관련 내용텐서차원분할까지많은개 GPU 상。에서 FLUX 중，주요응용에서 `ReplicatedLinear` 이이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer)`runtime/layers/linear.py`）：

```python
class ReplicatedLinear(nn.Module):
    """지원 TP 의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer)
    def forward(self, x):
        # 에서 TP 모드하，weight관련 내용분할
        output = F.linear(x, self.weight, self.bias)
        # All-reduce 관련 내용결과
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output
```

TP 의좋은관련 내용이다가능로줄인다이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용、관련 내용계산그리고row관련 내용대해큰모델관련 내용있다관련 내용

### 3.2 Ulysses Sequence Parallelism

Ulysses SP 이다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)그리고row관련 내용통해 all-to-all 관련 내용에서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)와관련 내용차원관련 내용수행한다분할（`UlyssesAttention` 의구현에서 `runtime/layers/attention/layer.py`）：

```python
class UlyssesAttention(nn.Module):
    def forward(self, q, k, v):
        # 입력: [B, S_local, H, D]
        
        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0)
        
        # All-to-all: 에서관련 내용와이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)차원이 부분은 원문의 해당 기술 설명을 이어서 서술한다
        # [3*B, S_local, H, D] -> [3*B, S_global, H_local, D]
        qkv = sequence_model_parallel_all_to_all_4D(
            qkv, scatter_dim=2, gather_dim=1
        )
        
        # 실행한다 attention
        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
        
        # All-to-all: 관련 내용원본관련 내용
        # [B, S_global, H_local, D] -> [B, S_local, H, D]
        output = sequence_model_parallel_all_to_all_4D(
            output, scatter_dim=1, gather_dim=2
        )
        
        return output
```

Ulysses SP 의관련 내용이다관련 내용의：입력단계각개 GPU 관련 내용있다완전한이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)의관련 내용부분와완전한의관련 내용통해 all-to-all 관련 내용할 것이다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)차원 gather、관련 내용차원 scatter，그다음각개 GPU 계산완전한이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)의부분관련 내용마지막으로다시통해 all-to-all 관련 내용원본관련 내용

### 3.3 USP (Unified Sequence Parallelism)

USP  Ulysses SP 와 Ring Attention 관련 내용와서（`USPAttention` 의구현도에서 `runtime/layers/attention/layer.py`）：

```python
class USPAttention(nn.Module):
    def forward(self, q, k, v):
        # Ulysses-style All-to-All
        if get_ulysses_parallel_world_size() > 1:
            q = _usp_input_all_to_all(q, head_dim=2)
            k = _usp_input_all_to_all(k, head_dim=2)
            v = _usp_input_all_to_all(v, head_dim=2)
        
        # Ring Attention（만약관련 내용사용）
        if get_ring_parallel_world_size() > 1:
            out = ring_attn(q, k, v, attn_impl=self.attn_impl)
        else:
            out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
        
        # Ulysses-style All-to-All（관련 내용
        if get_ulysses_parallel_world_size() > 1:
            out = _usp_output_all_to_all(out, head_dim=2)
        
        return out
```

USP  Ulysses 와 Ring 의이 부분은 원문의 해당 기술 설명을 이어서 서술한다와서，그리고row설정더관련 내용대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)있다사용。

### 3.4 CFG Parallelism

Classifier-Free Guidance (CFG) 그리고row이다관련 내용의계산할당까지아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (GPU)상：

```python
# 에서 DenoisingStage 중
if batch.do_classifier_free_guidance:
    # CFG rank 0 계산관련 내용
    # CFG rank 1 계산관련 내용
    cfg_rank = get_classifier_free_guidance_rank()
    
    if cfg_rank == 0:
        noise_pred = transformer(latents, pos_prompt_embeds,...)
    else:
        noise_pred = transformer(latents, neg_prompt_embeds,...)
    
    # All-gather 관련 내용결과
    noise_pred = cfg_model_parallel_all_gather(noise_pred, dim=0)
    
    # 관련 내용
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

### 3.5 그리고row관련 내용에서모델중의관련 내용사용

이들그리고row관련 내용에서아니관련 내용모델중의관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다필자는로관련 내용개관련 내용모델로관련 내용설명。

**FLUX 모델중의관련 내용사용**（`runtime/models/dits/flux.py`）：

```python
class FluxAttention(nn.Module):
    def __init__(self, query_dim, num_heads,...):
        # TP 지원：관련 내용사용 ReplicatedLinear
        self.to_q = ReplicatedLinear(query_dim, self.inner_dim, bias=bias)
        self.to_k = ReplicatedLinear(query_dim, self.inner_dim, bias=bias)
        self.to_v = ReplicatedLinear(query_dim, self.inner_dim, bias=bias)
        
        # 출력관련 내용도사용 ReplicatedLinear
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(
            ReplicatedLinear(self.inner_dim, self.out_dim, bias=out_bias)
        )
        
        # 관련 내용사용 USPAttention 지원이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)그리고row
        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
        )
```

FLUX 의관련 내용있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer)모두사용 `ReplicatedLinear` 관련 내용의 `nn.Linear`，관련 내용에서관련 내용사용 TP 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (weight)된다관련 내용분할。Attention layer관련 내용사용 `USPAttention`，가능로관련 내용지원 Ulysses 와 Ring 그리고row。

**HunyuanVideo 모델중의관련 내용사용**（`runtime/models/dits/hunyuanvideo.py`）：

```python
class MMDoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads,...):
        # QKV 관련 내용사용 ReplicatedLinear
        self.img_attn_qkv = ReplicatedLinear(
            hidden_size, hidden_size * 3, bias=qkv_bias
        )
        self.txt_attn_qkv = ReplicatedLinear(
            hidden_size, hidden_size * 3, bias=qkv_bias
        )
        
        # 관련 내용사용 UlyssesAttention
        self.attn = UlyssesAttention(
            num_heads=num_attention_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )
```

HunyuanVideo 관련 내용사용의이다 `UlyssesAttention` 관련 내용아니이다 `USPAttention`，왜냐하면관련 내용아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Ring Attention)이선택관련 내용모델의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)와그리고row관련 내용

**WanVideo 모델중의관련 내용사용**（`runtime/models/dits/wanvideo.py`）：

```python
class WanVideoSelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads,...):
        # QKV 관련 내용
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        
        # 관련 내용사용관련 내용의 UlyssesAttention_VSA（Video Sparse Attention）
        self.attn1 = UlyssesAttention_VSA(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends={
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            },
        )
```

WanVideo 관련 내용사용관련 내용의 `UlyssesAttention_VSA`，관련 내용이다대상으로관련 내용생성한다최적화의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (attention)

부터이들관련 내용가능로보다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SGLang Diffusion)의그리고row관련 내용사용관련 내용
- 관련 내용있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer)사용 `ReplicatedLinear` 이 부분은 원문의 해당 기술 설명을 이어서 서술한다지원 TP
- Attention layer관련 내용선택 `UlyssesAttention`、`USPAttention` 또는관련 내용
- 아니관련 내용전이 부분은 원문의 해당 기술 설명을 이어서 서술한다그리고row관련 내용에서이 부분은 원문의 해당 기술 설명을 이어서 서술한다

### 3.6 그리고row관련 내용설정

이들그리고row관련 내용모두가능로통해명령row파라미터와서설정：

```bash
# TP=2, Ulysses=2
sglang serve --model-path FLUX.1-dev \
    --tp-size 2 \
    --ulysses-degree 2 \
    --num-gpus 4

# USP: Ulysses=2, Ring=2
sglang serve --model-path FLUX.1-dev \
    --ulysses-degree 2 \
    --ring-degree 2 \
    --num-gpus 4

# CFG Parallel
sglang serve --model-path FLUX.1-dev \
    --enable-cfg-parallel \
    --num-gpus 2
```

## 0x4. Attention Backend 상세 해설

SGLang Diffusion 지원많은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (attention backend)가능로관련 내용와관련 내용선택관련 내용의구현。

### 4.1 Backend 선택관련 내용

Backend 선택관련 내용에서 `runtime/layers/attention/selector.py` 중：

```python
def get_attn_backend(head_size: int, dtype: torch.dtype, 
                     supported_attention_backends: set[AttentionBackendEnum]) -> AttentionBackend:
    # 관련 내용와관련 내용선택 backend
    backend_cls_str = current_platform.get_attn_backend_cls_str(
        selected_backend, head_size, dtype
    )
    
    # 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (backend)
    backend_cls = import_from_string(backend_cls_str)
    return backend_cls()
```

### 4.2 FlashAttention Backend

기본관련 내용사용의이다 FlashAttention 이높은성능 attention 구현（`runtime/layers/attention/backends/flash_attn.py`）：

```python
class FlashAttentionImpl(AttentionImpl):
    def forward(self, query, key, value, attn_metadata=None):
        output = flash_attn_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=query.shape[1],
            max_seqlen_k=key.shape[1],
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            ver=fa_ver,  # FA3 for Hopper, FA4 for Blackwell
        )
        return output
```

FlashAttention 관련 내용사용의이다 sgl-kernel 의최적화구현，지원 FA3（Hopper）와 FA4（Blackwell），관련 내용높은관련 내용도빠른。

### 4.3 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Backend)

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (FlashAttention SGLang Diffusion)지원이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (backend Torch SDPA PyTorch)구현，관련 내용좋은）、Sage Attention（대상으로이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)최적화）、Sliding Tile Attention（관련 내용생성한다）、Video Sparse Attention（이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (attention)줄인다계산이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (VMOBA Attention MoBA attention)

Backend 선택관련 내용

```python
# CUDA 관련 내용
if selected_backend == AttentionBackendEnum.FA:
    if is_blackwell():
        set_fa_ver(4)  # 관련 내용사용 FA4
    else:
        set_fa_ver(3)  # 관련 내용사용 FA3
    return FlashAttentionBackend
elif selected_backend == AttentionBackendEnum.SAGE_ATTN:
    return SageAttentionBackend
#... 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (backend)
```

### 4.4 Backend 설정

가능로통해관련 내용변수와서선택 backend：

```bash
# 관련 내용사용 FlashAttention
export SGLANG_DIFFUSION_ATTENTION_BACKEND=fa

# 관련 내용사용 Sage Attention
export SGLANG_DIFFUSION_ATTENTION_BACKEND=sage_attn

# 관련 내용사용 Torch SDPA
export SGLANG_DIFFUSION_ATTENTION_BACKEND=torch_sdpa
```

## 0x5. 추가새모델지원관련 내용

필자는관련 내용하추가새모델의관련 내용로 FLUX.1-dev 로관련 내용

```mermaid
graph TD
    A[관련 내용] --> B[관련 내용모델설정]
    B --> C[구현 Transformer 모델]
    C --> D[구현 Pipeline]
    D --> E[관련 내용모델]
    E --> F[설정그리고row관련 내용]
    F --> G[완료]
    
    B --> B1[configs/models/dits/flux.py<br/>FluxArchConfig, FluxConfig]
    B --> B2[configs/models/vaes/flux.py<br/>FluxVAEConfig]
    B --> B3[configs/pipeline_configs/flux.py<br/>FluxPipelineConfig]
    
    C --> C1[runtime/models/dits/flux.py<br/>FluxTransformer2DModel]
    C --> C2[이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CachableDiT)]
    C --> C3[구현 forward 관련 내용]
    
    D --> D1[runtime/pipelines/flux.py<br/>FluxPipeline]
    D --> D2[이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (ComposedPipelineBase)]
    D --> D3[create_pipeline_stages<br/>추가관련 내용개 stage]
    
    E --> E1[registry.py<br/>register_configs]
    
    F --> F1[TP 지원: ReplicatedLinear<br/>tensor_model_parallel_all_reduce]
    F --> F2[Ulysses SP: UlyssesAttention<br/>sequence_model_parallel_all_to_all]
    F --> F3[에서 Transformer Block 중<br/>관련 내용사용대응의 Attention layer]
```

관련 내용와서이 부분은 원문의 해당 기술 설명을 이어서 서술한다

1. 관련 내용모델설정
   - `configs/models/dits/flux.py`: 관련 내용`FluxArchConfig` 와 `FluxConfig`，관련 내용모델관련 내용파라미터
   - `configs/models/vaes/flux.py`: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (VAE)설정
   - `configs/pipeline_configs/flux.py`: 관련 내용`FluxPipelineConfig`，이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (DiT VAE Text Encoder)

2. 구현 Transformer 모델
   - `runtime/models/dits/flux.py`: 구현 `FluxTransformer2DModel`，관련 내용`CachableDiT`
   - 에서 `__init__` 중초기화이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer embedding transformer blocks)
   - 구현 `forward` 이 부분은 원문의 해당 기술 설명을 이어서 서술한다전이 부분은 원문의 해당 기술 설명을 이어서 서술한다

3. 구현 Pipeline
   - `runtime/pipelines/flux.py`: 구현 `FluxPipeline`，관련 내용`ComposedPipelineBase`
   - 에서 `create_pipeline_stages` 중추가관련 내용개 stage（TextEncodingStage、DenoisingStage 관련 내용
   - 각개 stage 통해 `self.get_module()` 얻는다대응의관련 내용

4. 관련 내용모델
   - `registry.py`: 호출한다 `register_configs` 할 것이다모델관련 내용와설정관련 내용

5. 설정그리고row관련 내용
   - TP 지원：에서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer)중관련 내용사용 `ReplicatedLinear`，에서전관련 내용후호출한다 `tensor_model_parallel_all_reduce`
   - Ulysses SP 지원：에서 Transformer Block 의 attention layer관련 내용사용 `UlyssesAttention` 또는 `USPAttention`
   - 에서모델초기화관련 내용`server_args.tp_size`、`server_args.ulysses_degree` 관련 내용파라미터설정그리고row

## 0x6. Profiler 관련 내용사용

SGLang Diffusion 관련 내용성능분석이 부분은 원문의 해당 기술 설명을 이어서 서술한다의성능관련 내용와관련 내용의 torch profiler。필자는관련 내용소개관련 내용하。

### 6.1 성능이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Performance Logger)

성능관련 내용`runtime/utils/perf_logger.py`）가능로기록각개 stage 와 denoising step 의관련 내용

```bash
# 관련 내용성능관련 내용목차
export SGLANG_PERF_LOG_DIR=/path/to/logs

# 시작관련 내용
sglang serve --model-path black-forest-labs/FLUX.1-dev --port 3000
```

성능관련 내용된다관련 내용기록까지 `SGLANG_PERF_LOG_DIR` 목차，관련 내용각개 stage 의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (TextEncodingStage DenoisingStage)각개 denoising step 의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Git commit hash)와관련 내용

관련 내용로 JSON，가능로관련 내용사용관련 내용분석：

```python
import json

with open('perf_log.json') as f:
    data = json.load(f)
    
print(f"Total duration: {data['total_duration_ms']:.2f}ms")
for stage, duration in data['stages'].items():
    print(f"{stage}: {duration:.2f}ms")
```

### 6.2 Torch Profiler

만약관련 내용더관련 내용의성능분석，가능로사용 torch profiler（구현에서 `DenoisingStage` 의 `start_profile`/`stop_profile` 관련 내용중）。torch profiler 가능로기록 CPU 와 GPU 의관련 내용실행한다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (operator)사용、kernel 호출한다관련 내용

관련 내용사용 torch profiler 의이 부분은 원문의 해당 기술 설명을 이어서 서술한다에서 `sglang generate` 명령중관련 내용`--profile` 파라미터관련 내용 (row)

```bash
# 관련 내용사용 --profile 관련 내용사용 profiler
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A cute baby sea otter" \
    --profile \
    --num-profiled-timesteps 8  # 가능관련 내용기록전 8 개 denoising step
```

Profiler 설정파라미터：

```python
# 에서 DenoisingStage 중의설정
self.profiler = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,  # 만약 CUDA 가능사용
    ],
    schedule=torch.profiler.schedule(
        skip_first=0,  # 아니이 부분은 원문의 해당 기술 설명을 이어서 서술한다
        wait=0,        # 아니관련 내용
        warmup=1,      # 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (1)
        active=batch.num_profiled_timesteps,  # 기록관련 내용개수의관련 내용
        repeat=5,      # 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (5)
    ),
    record_shapes=True,   # 기록텐서shape
    with_stack=True,      # 기록호출한다관련 내용
)
```

Profiler 출력：

생성한다의 trace 파일저장에서 `./logs` 목차하，관련 내용로 `{request_id}-rank{rank}.trace.json.gz`。가능로관련 내용사용 Chrome 의 `chrome://tracing` 또는 TensorBoard 관련 내용보다：

```bash
# 관련 내용사용 TensorBoard 관련 내용보다
tensorboard --logdir=./logs

# 또는관련 내용에서 Chrome 중켜다 trace 파일
# 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (chrome://tracing)그다음로드.trace.json.gz 파일
```

Profiler 의좋은관련 내용이다볼 수 있다각개 CUDA kernel 의실행한다관련 내용분석 CPU 와 GPU 관련 내용의동기화 overhead，관련 내용성능이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)시작overhead이 부분은 원문의 해당 기술 설명을 이어서 서술한다지원많은관련 내용분석，각개 rank 생성한다독립의 trace 파일。

다만관련 내용주의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Profiler)된다관련 내용의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row overhead)아니권장관련 내용대해관련 내용큰모델，권장만 profile 적은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (timesteps 3-8)만약관련 내용까지 OOM，가능로관련 내용`record_shapes=False` 와 `with_stack=False` 와서줄인다관련 내용사용。

## 0x7. 관련 내용사용예제

### 7.1 관련 내용

```bash
# Use the latest release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
pip install -e "python[diffusion]"

# With uv
uv pip install -e "python[diffusion]" --prerelease=allow
```

### 7.2 시작관련 내용

```bash
# 관련 내용
sglang serve --model-path black-forest-labs/FLUX.1-dev --port 3000

또는관련 내용

sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion"

# 많은관련 내용 (TP)
sglang serve --model-path black-forest-labs/FLUX.1-dev \
    --tp-size 2 --num-gpus 2 --port 3000

# Ulysses SP
sglang serve --model-path black-forest-labs/FLUX.1-dev \
    --ulysses-degree 2 --num-gpus 2 --port 3000
```

### 7.3 호출한다 API

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# 관련 내용
response = requests.post(
    "http://127.0.0.1:3000/v1/images/generations",
    headers={"Content-Type": "application/json"},
    json={
        "model": "black-forest-labs/FLUX.1-dev",
        "prompt": "A cute baby sea otter",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"
    }
)

# 관련 내용이미지
result = response.json()
image_data = base64.b64decode(result["data"][0]["b64_json"])
image = Image.open(BytesIO(image_data))
image.save("output.png")
```

관련 내용개log：

```shell
sglang serve --model-path black-forest-labs/FLUX.1-dev --port 3000

[12-05 09:17:00] Downloaded model to /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
[12-05 09:17:00] Model path: /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
[12-05 09:17:00] Diffusers version: 0.30.0.dev0
[12-05 09:17:00] Loading pipeline modules from config: {'_class_name': 'FluxPipeline', '_diffusers_version': '0.30.0.dev0', 'scheduler': ['diffusers', 'FlowMatchEulerDiscreteScheduler'], 'text_encoder': ['transformers', 'CLIPTextModel'], 'text_encoder_2': ['transformers', 'T5EncoderModel'], 'tokenizer': ['transformers', 'CLIPTokenizer'], 'tokenizer_2': ['transformers', 'T5TokenizerFast'], 'transformer': ['diffusers', 'FluxTransformer2DModel'], 'vae': ['diffusers', 'AutoencoderKL']}
[12-05 09:17:00] Loading required components: ['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'transformer', 'scheduler']
Loading required modules:   0%|                                                                                                                                                       | 0/7 [00:00<?,?it/s][12-05 09:17:00] Loading text_encoder using transformers from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder
[12-05 09:17:00] Loading text_encoder from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder
[12-05 09:17:00] HF model config: {'architectures': ['CLIPTextModel'], 'attention_dropout': 0.0, 'bos_token_id': 0, 'dropout': 0.0, 'eos_token_id': 2, 'hidden_act': 'quick_gelu', 'hidden_size': 768, 'initializer_factor': 1.0, 'initializer_range': 0.02, 'intermediate_size': 3072, 'layer_norm_eps': 1e-05, 'max_position_embeddings': 77, 'num_attention_heads': 12, 'num_hidden_layers': 12, 'pad_token_id': 1, 'projection_dim': 768, 'vocab_size': 49408}
[12-05 09:17:00] Using FlashAttention (FA3 for hopper, FA4 for blackwell) backend
[12-05 09:17:00] [RunAI Streamer] Overall time to stream 234.7 MiB of all files to cpu: 0.56s, 420.2 MiB/s
[12-05 09:17:00] Loading weights took 0.57 seconds
[12-05 09:17:01] Loaded text_encoder: FSDPCLIPTextModel from: customized
[12-05 09:17:01] Loaded module text_encoder from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder
Loading required modules:  14%|████████████████████▍                                                                                                                          | 1/7 [00:01<00:10,  1.68s/it][12-05 09:17:01] Loading text_encoder_2 using transformers from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder_2
[12-05 09:17:01] Loading text_encoder_2 from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder_2
[12-05 09:17:01] HF model config: {'architectures': ['T5EncoderModel'], 'classifier_dropout': 0.0, 'd_ff': 10240, 'd_kv': 64, 'd_model': 4096, 'decoder_start_token_id': 0, 'dense_act_fn': 'gelu_new', 'dropout_rate': 0.1, 'eos_token_id': 1, 'feed_forward_proj': 'gated-gelu', 'initializer_factor': 1.0, 'is_encoder_decoder': True, 'is_gated_act': True, 'layer_norm_epsilon': 1e-06, 'num_decoder_layers': 24, 'num_heads': 64, 'num_layers': 24, 'output_past': True, 'pad_token_id': 0, 'relative_attention_max_distance': 128, 'relative_attention_num_buckets': 32, 'tie_word_embeddings': False, 'use_cache': True, 'vocab_size': 32128}
[12-05 09:17:07] [RunAI Streamer] Overall time to stream 8.9 GiB of all files to cpu: 5.4s, 1.6 GiB/s
[12-05 09:17:07] Loading weights took 5.45 seconds
[12-05 09:17:27] Loaded text_encoder_2: FSDPT5EncoderModel from: customized
[12-05 09:17:27] Loaded module text_encoder_2 from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder_2
Loading required modules:  29%|████████████████████████████████████████▊                                                                                                      | 2/7 [00:27<01:19, 15.89s/it][12-05 09:17:27] Loading tokenizer using transformers from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer
[12-05 09:17:27] Loading tokenizer from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer
[12-05 09:17:27] Loaded tokenizer: CLIPTokenizerFast from: customized
[12-05 09:17:27] Loaded module tokenizer from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer
[12-05 09:17:27] Loading tokenizer_2 using transformers from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer_2
[12-05 09:17:27] Loading tokenizer_2 from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer_2
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
[12-05 09:17:27] Loaded tokenizer_2: T5TokenizerFast from: customized
[12-05 09:17:27] Loaded module tokenizer_2 from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer_2
Loading required modules:  57%|█████████████████████████████████████████████████████████████████████████████████▋                                                             | 4/7 [00:27<00:18,  6.01s/it][12-05 09:17:27] Loading vae using diffusers from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae
[12-05 09:17:27] Loading vae from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae
[12-05 09:17:27] HF model config: {'_name_or_path': '../checkpoints/flux-dev', 'act_fn': 'silu', 'block_out_channels': [128, 256, 512, 512], 'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'], 'force_upcast': True, 'in_channels': 3, 'latent_channels': 16, 'latents_mean': None, 'latents_std': None, 'layers_per_block': 2, 'mid_block_add_attention': True, 'norm_num_groups': 32, 'out_channels': 3, 'sample_size': 1024, 'scaling_factor': 0.3611, 'shift_factor': 0.1159, 'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], 'use_post_quant_conv': False, 'use_quant_conv': False}
[12-05 09:17:28] Loaded vae: AutoencoderKL from: customized
[12-05 09:17:28] Loaded module vae from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae
Loading required modules:  71%|██████████████████████████████████████████████████████████████████████████████████████████████████████▏                                        | 5/7 [00:28<00:08,  4.34s/it][12-05 09:17:28] Loading transformer using diffusers from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer
[12-05 09:17:28] Loading transformer from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer
[12-05 09:17:28] transformer cls_name: FluxTransformer2DModel
[12-05 09:17:28] Loading model from 3 safetensors files: ['/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00001-of-00003.safetensors', '/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00002-of-00003.safetensors', '/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00003-of-00003.safetensors']
[12-05 09:17:28] Loading FluxTransformer2DModel, default_dtype: torch.bfloat16
[12-05 09:17:28] Using FlashAttention (FA3 for hopper, FA4 for blackwell) backend
[12-05 09:17:39] [RunAI Streamer] Overall time to stream 22.2 GiB of all files to cpu: 11.36s, 2.0 GiB/s
[12-05 09:17:46] Loaded model with 11.90B parameters
[12-05 09:17:46] Loaded transformer: FluxTransformer2DModel from: customized
[12-05 09:17:46] Loaded module transformer from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer
Loading required modules:  86%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                    | 6/7 [00:46<00:08,  8.42s/it][12-05 09:17:46] Loading scheduler using diffusers from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/scheduler
[12-05 09:17:46] Loading scheduler from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/scheduler
[12-05 09:17:46] Loaded scheduler: FlowMatchEulerDiscreteScheduler from: customized
[12-05 09:17:46] Loaded module scheduler from /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/scheduler
Loading required modules: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:46<00:00,  6.58s/it]
[12-05 09:17:46] Pipelines instantiated
[12-05 09:17:46] Worker 0: Initialized device, model, and distributed environment.
[12-05 09:17:46] Worker 0: Scheduler loop started.
[12-05 09:17:46] Rank 0 scheduler listening on tcp://*:5592
[12-05 09:17:46] Starting FastAPI server.
[12-05 09:17:46] Started server process [37013]
[12-05 09:17:46] Waiting for application startup.
[12-05 09:17:46] Scheduler client connected to backend scheduler at tcp://localhost:5592
[12-05 09:17:46] ZMQ Broker is listening for offline jobs on tcp://*:3001
[12-05 09:17:46] Application startup complete.
[12-05 09:17:46] Uvicorn running on http://localhost:3000 (Press CTRL+C to quit)
[12-05 09:19:06] 127.0.0.1:51806 - "GET /metrics HTTP/1.1" 404
[12-05 09:21:32] Sampling params:
                       width: 1024
                      height: 1024
                  num_frames: 1
                      prompt: A cute baby sea otter
                  neg_prompt: None
                        seed: 1024
                 infer_steps: 50
      num_outputs_per_prompt: 1
              guidance_scale: 1.0
     embedded_guidance_scale: 3.5
                    n_tokens: 16384
                  flow_shift: None
                  image_path: None
                 save_output: True
            output_file_path: outputs/8c8083c6-870e-4f30-b682-15fdc2f58910.jpg
        
[12-05 09:21:32] Processing prompt: A cute baby sea otter
[12-05 09:21:32] Creating pipeline stages...
[12-05 09:21:32] Using FlashAttention (FA3 for hopper, FA4 for blackwell) backend
[12-05 09:21:32] Running pipeline stages: ['input_validation_stage', 'prompt_encoding_stage_primary', 'conditioning_stage', 'timestep_preparation_stage', 'latent_preparation_stage', 'denoising_stage', 'decoding_stage']
[12-05 09:21:32] [InputValidationStage] started...
[12-05 09:21:32] [InputValidationStage] finished in 0.0003 seconds
[12-05 09:21:32] [TextEncodingStage] started...
[12-05 09:21:33] Running FA4 warmup (global/causal/local, LSE on/off, optional GQA pack)...
[12-05 09:21:51] [TextEncodingStage] finished in 19.1504 seconds
[12-05 09:21:51] [ConditioningStage] started...
[12-05 09:21:51] [ConditioningStage] finished in 0.0001 seconds
[12-05 09:21:51] [TimestepPreparationStage] started...
[12-05 09:21:51] [TimestepPreparationStage] finished in 0.0924 seconds
[12-05 09:21:51] [LatentPreparationStage] started...
[12-05 09:21:51] [LatentPreparationStage] finished in 0.0005 seconds
[12-05 09:21:51] [DenoisingStage] started...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:09<00:00,  5.55it/s]
[12-05 09:22:01] [DenoisingStage] average time per step: 0.1804 seconds
[12-05 09:22:01] [DenoisingStage] finished in 9.0389 seconds
[12-05 09:22:01] [DecodingStage] started...
[12-05 09:22:02] [DecodingStage] finished in 1.8255 seconds
[12-05 09:22:03] Saved output to outputs/8c8083c6-870e-4f30-b682-15fdc2f58910.jpg
[12-05 09:22:03] Pixel data generated successfully in 30.35 seconds
[12-05 09:22:03] Completed batch processing. Generated 1 outputs in 30.35 seconds.
[12-05 09:22:03] 127.0.0.1:46656 - "POST /v1/images/generations HTTP/1.1" 200
```

이이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (FLUX.1-dev)모델부터시작까지생성한다관련 내용이미지의완전한관련 내용필자는관련 내용와서관련 내용하관련 내용개단계：

**1. 관련 내용시작와초기화（09:14:31 - 09:14:38）**

먼저관련 내용`server_args`，볼 수 있다핵심설정：
- `num_gpus=1, tp_size=1`：이 부분은 원문의 해당 기술 설명을 이어서 서술한다있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (TP)
- `ulysses_degree=1, ring_degree=1`：관련 내용있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (column)그리고row
- `dit_cpu_offload=true, text_encoder_cpu_offload=true, vae_cpu_offload=true`：DiT、Text Encoder 와 VAE 모두이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CPU offload)이다위해관련 내용

그다음초기화이 부분은 원문의 해당 기술 설명을 이어서 서술한다만있다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SGLang)이다된다초기화 Gloo 관련 내용위해코드관련 내용

**2. 모델로드단계（09:14:41 - 09:17:46）**

관련 내용부분대응 `ComposedPipelineBase.load_modules` 관련 내용의실행한다：

```python
# 읽기 model_index.json，관련 내용이다 FluxPipeline
[12-05 09:14:41] Downloaded model_index.json for black-forest-labs/FLUX.1-dev, pipeline: FluxPipeline
```

그다음관련 내용로드 7 개관련 내용대응 `FluxPipelineConfig._required_config_modules`）：

- **text_encoder (CLIP)**：로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (1.68s 234.7 MiB)사용 FlashAttention backend
- **text_encoder_2 (T5)**：로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (15.89s 8.9 GiB)이다관련 내용큰의관련 내용
- **tokenizer & tokenizer_2**：로드관련 내용빠른，만이다설정파일
- **vae**：로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (1s 168 MiB)
- **transformer (DiT)**：로드이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (18s 22.2 GiB 11.90B)파라미터，관련 내용이다관련 내용모델
- **scheduler**：로드설정

관련 내용개로드관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (3)주요관련 내용에서하관련 내용와로드 transformer 와 text_encoder_2 상。

**3. Pipeline 생성한다（09:17:46）**

```python
[12-05 09:17:46] Pipelines instantiated
[12-05 09:17:46] Worker 0: Initialized device, model, and distributed environment.
```

여기호출한다 `FluxPipeline.create_pipeline_stages`，생성한다 7 개 stage：InputValidationStage、TextEncodingStage、ConditioningStage、TimestepPreparationStage、LatentPreparationStage、DenoisingStage、DecodingStage。

**4. 관련 내용단계（09:21:32 - 09:22:03）**

관련 내용까지관련 내용후，관련 내용실행한다관련 내용개 stage：

```python
# 관련 내용파라미터
width=1024, height=1024, infer_steps=50, guidance_scale=1.0

# 관련 내용개 stage 의관련 내용
[InputValidationStage] 0.0003s        # 검증입력파라미터
[TextEncodingStage] 19.1504s          # CLIP + T5 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (FA4 warmup)
[ConditioningStage] 0.0001s           # 관련 내용
[TimestepPreparationStage] 0.0924s    # 이 부분은 원문의 해당 기술 설명을 이어서 서술한다
[LatentPreparationStage] 0.0005s      # 초기화 latent
[DenoisingStage] 9.0389s              # 50 관련 내용가서관련 내용각이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (0.1804s)
[DecodingStage] 1.8255s               # VAE 관련 내용
```

볼 수 있다 TextEncodingStage 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (19.15s)이다왜냐하면：
1. 제이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row warmup FlashAttention 4)`Running FA4 warmup`）
2. T5 모델관련 내용큰（8.9 GiB），관련 내용느린
3. 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CPU offload)에서 CPU 와 GPU 이 부분은 원문의 해당 기술 설명을 이어서 서술한다

DenoisingStage 이다제관련 내용의（9.04s），관련 내용이다관련 내용의가서관련 내용실행한다 50 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (transformer)전관련 내용

**5. 관련 내용분석**

```python
[12-05 09:22:03] Pixel data generated successfully in 30.35 seconds
```

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (30.35)여기서：
- TextEncodingStage: 19.15s (63%)
- DenoisingStage: 9.04s (30%)
- DecodingStage: 1.83s (6%)
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (stage: < 0.1s)

만약이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (CPU offload)또는관련 내용사용많은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (TP/SP)성능된다있다관련 내용향상。이관련 내용좋은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SGLang Diffusion)의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (block)각개 stage 의관련 내용모두관련 내용기록하와서，관련 내용성능분석와최적화。

### 7.4 명령row생성한다

```bash
# 관련 내용생성한다이미지
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --save-output
```



## 0x8. 정리

이 글기록필자는대해 SGLang Diffusion 관련 내용의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (SGLang Diffusion)통해 ComposedPipelineBase + PipelineStage 의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (block)개관련 내용의관련 내용모델이 부분은 원문의 해당 기술 설명을 이어서 서술한다지원관련 내용의그리고row이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (TP Ulysses SP USP CFG Parallel)와많은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Attention Backend FlashAttention Sage Attention)가능로높은관련 내용배포관련 내용모델。추가새모델의관련 내용도관련 내용만관련 내용구현설정이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Transformer)모델、Pipeline 관련 내용그리고이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row)통해 ReplicatedLinear 와 UlyssesAttention 관련 내용가능로관련 내용지원그리고row。

## 참고 자료

- SGLang Diffusion 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (:)https://lmsys.org/blog/2025-11-07-sglang-diffusion/
- SGLang GitHub: https://github.com/sgl-project/sglang
- FastVideo: https://github.com/hao-ai-lab/FastVideo
- FLUX.1 모델: https://huggingface.co/black-forest-labs/FLUX.1-dev
