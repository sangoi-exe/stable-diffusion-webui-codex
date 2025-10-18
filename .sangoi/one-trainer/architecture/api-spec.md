**Core Types**
- `EngineCapabilities`: `engine_id`, `model_types`, `tasks` (`{"txt2img","img2img","txt2vid","img2vid"}`), `dtype_support`, `devices`.
- `ProgressEvent`: `stage`, `percent`, `eta`, `step`, `total_steps`, `message`.
- Requests (dataclasses):
  - `Txt2ImgRequest`: `prompt`, `negative`, `width`, `height`, `steps`, `cfg`, `sampler`, `scheduler`, `seed`, `vae`, `clip_skip`, `lora`, `hires`, `batch`, `extras`.
  - `Img2ImgRequest`: campos do `Txt2Img` + `init_image`, `denoise_strength`, `mask` (opcional).
  - `Txt2VidRequest`: `prompt`, `negative`, `num_frames`, `fps`, `width`, `height`, `sampler`, `scheduler`, `seed`, `motion_strength`, `extras`.
  - `Img2VidRequest`: campos de vídeo + `init_image`/`frames`.

**Engine Interface (Python)**
```
class BaseInferenceEngine(ABC):
    engine_id: str

    @abstractmethod
    def capabilities(self) -> EngineCapabilities: ...

    @abstractmethod
    def load(self, model_ref: str, **options) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    # Optional tasks (raise NotImplementedError if unsupported)
    def txt2img(self, req: Txt2ImgRequest) -> Iterator[Union[ProgressEvent, ImageBatch]]: ...
    def img2img(self, req: Img2ImgRequest) -> Iterator[Union[ProgressEvent, ImageBatch]]: ...
    def txt2vid(self, req: Txt2VidRequest) -> Iterator[Union[ProgressEvent, VideoChunk]]: ...
    def img2vid(self, req: Img2VidRequest) -> Iterator[Union[ProgressEvent, VideoChunk]]: ...

    # Introspection
    def status(self) -> Dict[str, Any]: ...
    def memory_usage(self) -> Dict[str, Any]: ...
```

**Registry**
```
register_engine(key: str, cls: Type[BaseInferenceEngine]) -> None
create_engine(key: str, **kw) -> BaseInferenceEngine
list_engines() -> List[str]
resolve_for_model(model_ref: str) -> str  # heurística/metadata
```

**Orchestrator**
```
run(task: Literal["txt2img","img2img","txt2vid","img2vid"],
    model_key: str,
    request: Union[Txt2ImgRequest, Img2ImgRequest, Txt2VidRequest, Img2VidRequest]
) -> Iterator[Event]
```

**Events**
- `ProgressEvent` (sempre) e `ResultEvent` (`ImageBatch` ou `VideoChunk`) com metadados.

**Errors**
- `UnsupportedTaskError`, `InvalidPresetError`, `EngineLoadError`, `OOMError`, com mensagens acionáveis e código.

