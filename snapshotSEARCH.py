# snapshotOT.py
import os
import re
import io
from datetime import datetime
from pathlib import Path

try:
    from tkinter import Tk, filedialog
except ImportError:
    Tk = None

# Modo de execução: "TREE" para snapshot, "SEARCH" para busca de termo
mode = "SEARCH"  # Hardcoded: defina "TREE" ou "SEARCH"
search_term = "subseed"  # Hardcoded: termo a buscar nos arquivos

include_extensions = {".py"}

ignore_dirs = {
    ".git",
    ".idea",
    ".vscode",
    "dist",
    "node_modules",
    "test",
    "__pycache__",
    "cache",
    "workspace-cache",
    "venv",
    ".venv",
    "docs",
    "training_presets",    
    # "external",
    # "resources",
    # # Diretórios específicos de outros modelos
    # "modules/model/wuerstchen",
    # "modules/model/pixartAlpha",
    # "modules/model/stableDiffusion",
    # "modules/model/stableDiffusion3",
    # "modules/model/flux",
    # "modules/model/sana",
    # "modules/model/hunyuanVideo",
    # "modules/modelLoader/wuerstchen",
    # "modules/modelLoader/pixartAlpha",
    # "modules/modelLoader/stableDiffusion",
    # "modules/modelLoader/stableDiffusion3",
    # "modules/modelLoader/flux",
    # "modules/modelLoader/sana",
    # "modules/modelLoader/hunyuanVideo",
    # "modules/modelSaver/wuerstchen",
    # "modules/modelSaver/pixartAlpha",
    # "modules/modelSaver/stableDiffusion",
    # "modules/modelSaver/stableDiffusion3",
    # "modules/modelSaver/flux",
    # "modules/modelSaver/sana",
    # "modules/modelSaver/hunyuanVideo",
    # "modules/modelSetup/wuerstchen",
    # "modules/modelSetup/pixartAlpha",
    # "modules/modelSetup/stableDiffusion",
    # "modules/modelSetup/stableDiffusion3",
    # "modules/modelSetup/flux",
    # "modules/modelSetup/sana",
    # "modules/modelSetup/hunyuanVideo",
    # "modules/dataLoader/wuerstchen",
    # "modules/dataLoader/pixartAlpha",
    # "modules/dataLoader/stableDiffusion",
    # "modules/dataLoader/stableDiffusion3",
    # "modules/dataLoader/flux",
    # "modules/dataLoader/sana",
    # "modules/dataLoader/hunyuanVideo",
    # "modules/modelSampler/wuerstchen",
    # "modules/modelSampler/pixartAlpha",
    # "modules/modelSampler/stableDiffusion",
    # "modules/modelSampler/stableDiffusion3",
    # "modules/modelSampler/flux",
    # "modules/modelSampler/sana",
    # "modules/modelSampler/hunyuanVideo",
    # # Outros diretórios
    # "modules/cloud",
    # "scripts",
    # "embedding_templates",
    # "zluda",
    # "modules/module/quantized",
    # "modules/ui",
    # "training_concepts",
    # "training_deltas",
    # "training_presets",
    # "training_samples",
    # "modules/util/convert",
}

ignore_files = {
    "snapshotOT.py",
    "limpaLayer.py",
    "config.json",
    "full.txt",
    "filtered_layers.txt",
    "secrets.json",
    ".gitignore",
    ".gitattributes",
    "FluxBaseDataLoader.py",
    "HunyuanVideoBaseDataLoader.py",
    "PixArtAlphaBaseDataLoader.py",
    "SanaBaseDataLoader.py",
    "StableDiffusion3BaseDataLoader.py",
    "StableDiffusionBaseDataLoader.py",
    "StableDiffusionFineTuneVaeDataLoader.py",
    "WuerstchenBaseDataLoader.py",
    "FluxModel.py",
    "HunyuanVideoModel.py",
    "PixArtAlphaModel.py",
    "SanaModel.py",
    "StableDiffusion3Model.py",
    "StableDiffusionModel.py",
    "WuerstchenModel.py",
    "FluxEmbeddingModelLoader.py",
    "FluxFineTuneModelLoader.py",
    "FluxLoRAModelLoader.py",
    "HunyuanVideoEmbeddingModelLoader.py",
    "HunyuanVideoFineTuneModelLoader.py",
    "HunyuanVideoLoRAModelLoader.py",
    "PixArtAlphaEmbeddingModelLoader.py",
    "PixArtAlphaFineTuneModelLoader.py",
    "PixArtAlphaLoRAModelLoader.py",
    "SanaEmbeddingModelLoader.py",
    "SanaFineTuneModelLoader.py",
    "SanaLoRAModelLoader.py",
    "StableDiffusion3EmbeddingModelLoader.py",
    "StableDiffusion3FineTuneModelLoader.py",
    "StableDiffusion3LoRAModelLoader.py",
    "StableDiffusionEmbeddingModelLoader.py",
    "StableDiffusionFineTuneModelLoader.py",
    "StableDiffusionLoRAModelLoader.py",
    "WuerstchenEmbeddingModelLoader.py",
    "WuerstchenFineTuneModelLoader.py",
    "WuerstchenLoRAModelLoader.py",
    "FluxEmbeddingModelSaver.py",
    "FluxFineTuneModelSaver.py",
    "FluxLoRAModelSaver.py",
    "HunyuanVideoEmbeddingModelSaver.py",
    "HunyuanVideoFineTuneModelSaver.py",
    "HunyuanVideoLoRAModelSaver.py",
    "PixArtAlphaEmbeddingModelSaver.py",
    "PixArtAlphaFineTuneModelSaver.py",
    "PixArtAlphaLoRAModelSaver.py",
    "SanaEmbeddingModelSaver.py",
    "SanaFineTuneModelSaver.py",
    "SanaLoRAModelSaver.py",
    "StableDiffusion3EmbeddingModelSaver.py",
    "StableDiffusion3FineTuneModelSaver.py",
    "StableDiffusion3LoRAModelSaver.py",
    "StableDiffusionEmbeddingModelSaver.py",
    "StableDiffusionFineTuneModelSaver.py",
    "StableDiffusionLoRAModelSaver.py",
    "WuerstchenEmbeddingModelSaver.py",
    "WuerstchenFineTuneModelSaver.py",
    "WuerstchenLoRAModelSaver.py",
    "BaseFluxSetup.py",
    "FluxEmbeddingSetup.py",
    "FluxFineTuneSetup.py",
    "FluxLoRASetup.py",
    "BaseHunyuanVideoSetup.py",
    "HunyuanVideoEmbeddingSetup.py",
    "HunyuanVideoFineTuneSetup.py",
    "HunyuanVideoLoRASetup.py",
    "BasePixArtAlphaSetup.py",
    "PixArtAlphaEmbeddingSetup.py",
    "PixArtAlphaFineTuneSetup.py",
    "PixArtAlphaLoRASetup.py",
    "BaseSanaSetup.py",
    "SanaEmbeddingSetup.py",
    "SanaFineTuneSetup.py",
    "SanaLoRASetup.py",
    "BaseStableDiffusion3Setup.py",
    "StableDiffusion3EmbeddingSetup.py",
    "StableDiffusion3FineTuneSetup.py",
    "StableDiffusion3LoRASetup.py",
    "BaseStableDiffusionSetup.py",
    "StableDiffusionEmbeddingSetup.py",
    "StableDiffusionFineTuneSetup.py",
    "StableDiffusionFineTuneVaeSetup.py",
    "StableDiffusionLoRASetup.py",
    "BaseWuerstchenSetup.py",
    "WuerstchenEmbeddingSetup.py",
    "WuerstchenFineTuneSetup.py",
    "WuerstchenLoRASetup.py",
    "FluxSampler.py",
    "HunyuanVideoSampler.py",
    "PixArtAlphaSampler.py",
    "SanaSampler.py",
    "StableDiffusion3Sampler.py",
    "StableDiffusionSampler.py",
    "StableDiffusionVaeSampler.py",
    "WuerstchenSampler.py",
    "AestheticScoreModel.py",
    "BaseImageCaptionModel.py",
    "BaseImageMaskModel.py",
    "BaseRembgModel.py",
    "Blip2Model.py",
    "BlipModel.py",
    "ClipSegModel.py",
    "GenerateLossesModel.py",
    "HPSv2ScoreModel.py",
    "MaskByColor.py",
    "RembgHumanModel.py",
    "RembgModel.py",
    "WDModel.py",
}

ignore_file_patterns = [
    re.compile(r".*\.spec\.(js|py)$"),
    re.compile(r".*\.min\.(js|css)$"),
    re.compile(r".*test.*\.py$"),
    re.compile(r".*\.pyc$"),
    re.compile(r".*\.log$"),
    re.compile(r".*\.bak$"),
    re.compile(r".*\.tmp$"),
    re.compile(r".*\.swp$"),
    re.compile(r"\.DS_Store$"),
]


def optimize_content(content, ext):
    # Sua função optimize_content (mantida como está)
    if ext in {".js", ".css"}:
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        lines = [
            l.rstrip() for l in content.splitlines() if not l.lstrip().startswith("//")
        ]
    elif ext == ".py":
        lines = []
        in_multiline_comment = False
        for l in content.splitlines():
            stripped_l = l.lstrip()
            if stripped_l.startswith('"""') or stripped_l.startswith("'''"):
                quote_count = stripped_l.count('"""') + stripped_l.count("'''")
                if quote_count % 2 != 0:
                    in_multiline_comment = not in_multiline_comment
                    if (
                        quote_count == 2
                        and stripped_l.endswith(('"""', "'''"))
                        and len(stripped_l) > 5
                    ):
                        continue
                if in_multiline_comment and not stripped_l.endswith(('"""', "'''")):
                    continue
                elif not in_multiline_comment and stripped_l.startswith(('"""', "'''")):
                    pass
                elif quote_count >= 2 and stripped_l.endswith(('"""', "'''")):
                    continue
            if in_multiline_comment:
                if stripped_l.endswith('"""') or stripped_l.endswith("'''"):
                    in_multiline_comment = False
                continue
            if (
                not stripped_l.startswith("#")
                or stripped_l.startswith("# type:")
                or stripped_l.startswith("# noqa")
                or stripped_l.startswith("# pylint:")
            ):
                lines.append(l.rstrip())
    elif ext == ".handlebars":
        content = re.sub(r"{{!\s*.*?\s*}}", "", content)
        lines = [l.rstrip() for l in content.splitlines()]
    else:
        lines = [l.rstrip() for l in content.splitlines()]

    optimized, prev_empty = [], False
    for l in lines:
        if not l.strip():
            if not prev_empty:
                optimized.append("")
            prev_empty = True
        else:
            optimized.append(l)
            prev_empty = False
    if optimized and not optimized[-1].strip():
        optimized.pop()
    return "\n".join(optimized)


def should_include_file(relative_path: Path):
    """
    Verifica se um arquivo deve ser incluído no snapshot.
    """
    file_name = relative_path.name
    path_parts = {part for part in relative_path.parts}
    if any(ignored in path_parts for ignored in ignore_dirs):
        return False
    for ignored_dir_pattern in ignore_dirs:
        ignored_path = Path(ignored_dir_pattern)
        if ignored_path.parts == relative_path.parts[: len(ignored_path.parts)]:
            return False
    if file_name in ignore_files:
        return False
    file_ext_lower = relative_path.suffix.lower()
    if not file_ext_lower or file_ext_lower not in include_extensions:
        return False
    if any(p.match(file_name) for p in ignore_file_patterns):
        return False
    return True


def tree(
    root_path: Path,
    current_rel_path: Path,
    pad: str,
    out: io.StringIO,
    print_files: bool,
):
    """
    Percorre recursivamente a árvore de diretórios, aplicando filtros
    e escrevendo a estrutura e o conteúdo dos arquivos selecionados.
    """
    full_path = root_path / current_rel_path
    try:
        items = sorted(
            list(full_path.iterdir()), key=lambda p: (p.is_file(), p.name.lower())
        )
    except OSError as e:
        if any(ignored in current_rel_path.parts for ignored in ignore_dirs):
            return
        for ignored_dir_pattern in ignore_dirs:
            ignored_path = Path(ignored_dir_pattern)
            if ignored_path.parts == current_rel_path.parts[: len(ignored_path.parts)]:
                return
        out.write(f"{pad}+-- [Erro ao listar {current_rel_path.as_posix()}: {e}]\n")
        return

    dirs_to_process = []
    files_to_process = []

    for item_path in items:
        relative_item_path = current_rel_path / item_path.name
        is_in_ignored_dir = False
        temp_path = relative_item_path.parent
        while temp_path != Path("."):
            if temp_path.as_posix() in ignore_dirs or temp_path.name in ignore_dirs:
                is_in_ignored_dir = True
                break
            for ignored_dir_pattern in ignore_dirs:
                ignored_path = Path(ignored_dir_pattern)
                if ignored_path.parts == temp_path.parts[: len(ignored_path.parts)]:
                    is_in_ignored_dir = True
                    break
            if is_in_ignored_dir:
                break
            temp_path = temp_path.parent
        if is_in_ignored_dir:
            continue

        if item_path.is_dir():
            if (
                item_path.name not in ignore_dirs
                and relative_item_path.as_posix() not in ignore_dirs
            ):
                dirs_to_process.append(item_path.name)
        elif item_path.is_file():
            if should_include_file(relative_item_path):
                files_to_process.append(item_path.name)

    for f_name in files_to_process:
        relative_file_path = current_rel_path / f_name
        out.write(f"{pad}+-- {relative_file_path.as_posix()}\n")
        if print_files:
            try:
                with open(
                    full_path / f_name, "r", encoding="utf-8", errors="replace"
                ) as fc:
                    content = fc.read()
                ext = os.path.splitext(f_name)[1].lower()
                code_pad = pad + "    "
                out.write(
                    f"{code_pad}```{ext.lstrip('.')} linenums=\"1\"\n{optimize_content(content, ext)}\n{code_pad}```\n\n"
                )
            except Exception as e:
                out.write(f"{pad}    [Erro ao ler {f_name}: {e}]\n\n")

    for d_name in dirs_to_process:
        new_rel_path = current_rel_path / d_name
        out.write(f"{pad}+-- {new_rel_path.as_posix()}/\n")
        tree(root_path, new_rel_path, pad + "    ", out, print_files)

def collect_matching_files(root_path: Path, term: str) -> list[Path]:
    """
    Retorna uma lista de Paths de arquivos que:
    - passam no should_include_file
    - contêm 'term' em seu conteúdo
    """
    matches = []
    for full_path in root_path.rglob("*"):
        if full_path.is_file():
            rel = full_path.relative_to(root_path)
            if should_include_file(rel):
                try:
                    text = full_path.read_text(encoding="utf-8", errors="ignore")
                    if term in text:
                        matches.append(full_path)
                except Exception:
                    continue
    return matches

def snapshot_files(root_path: Path, files: list[Path], out: io.StringIO):
    """
    Escreve no buffer 'out' a estrutura e conteúdo dos 'files' listados,
    no mesmo formato de snapshot detalhado.
    """
    for full_path in sorted(files):
        rel = full_path.relative_to(root_path)
        out.write(f"+-- {rel.as_posix()}\n")
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            ext = full_path.suffix.lower()
            out.write(f"```{ext.lstrip('.')} linenums=\"1\"\n")
            out.write(optimize_content(content, ext))
            out.write("\n```\n\n")
        except Exception as e:
            out.write(f"    [Erro ao ler {rel.as_posix()}: {e}]\n\n")
# <<< Fim da adição da função de snapshot de arquivos >>>

# <<< Início da adição da função de busca >>>
def search_tree(root_path: Path, search_term: str) -> list[str]:
    """
    Percorre recursivamente 'root_path' filtrando arquivos via should_include_file
    e retorna ocorrências de 'search_term' no formato:
    'caminho/relativo:linha: conteúdo'.
    """
    matches: list[str] = []
    for full_path in root_path.rglob("*"):
        if full_path.is_file():
            rel = full_path.relative_to(root_path)
            if should_include_file(rel):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        for lineno, line in enumerate(f, 1):
                            if search_term in line:
                                matches.append(
                                    f"{rel.as_posix()}:{lineno}: {line.strip()}"
                                )
                except Exception:
                    continue
    return matches


# <<< Fim da adição da função de busca >>>

if __name__ == "__main__":
    # seleção de pasta (igual ao seu original)
    if Tk:
        Tk().withdraw()
        project_dir_str = filedialog.askdirectory(title="Selecione a pasta do projeto") or os.getcwd()
    else:
        project_dir_str = os.getcwd()

    project_dir = Path(project_dir_str)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = project_dir.name

    if mode == "TREE":
        # snapshot completo (igual ao original)
        simple_tree_buffer = io.StringIO()
        tree(project_dir, Path(""), "", simple_tree_buffer, False)
        dir_tree = simple_tree_buffer.getvalue()

        detailed_buffer = io.StringIO()
        tree(project_dir, Path(""), "", detailed_buffer, True)
        detailed = detailed_buffer.getvalue()

        header = (
            f"# Snapshot do Projeto (Foco em SDXL e Core)\n"
            f"Timestamp: {timestamp}\n\n"
            f"## Estrutura do Projeto:\n{dir_tree}\n"
            f"## Conteúdo do Projeto:\n"
        )
        final = header + detailed
        output_file = f"snapshot_SDXL_Core_{folder_name}_{timestamp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final)
        print(f"Snapshot salvo em {output_file}")

    elif mode == "SEARCH":
        # coleta arquivos que têm o termo
        matched = collect_matching_files(project_dir, search_term)
        # monta header de snapshot
        header = (
            f"# Snapshot de arquivos contendo '{search_term}'\n"
            f"Timestamp: {timestamp}\n\n"
            f"## Arquivos encontrados ({len(matched)}):\n"
        )
        # gera snapshot apenas desses arquivos
        buffer = io.StringIO()
        buffer.write(header)
        snapshot_files(project_dir, matched, buffer)

        output_file = f"snapshot_SEARCH_{folder_name}_{timestamp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(buffer.getvalue())
        print(f"Snapshot de SEARCH salvo em {output_file}")

    else:
        print(f"Modo desconhecido: {mode}. Use 'TREE' ou 'SEARCH'.")