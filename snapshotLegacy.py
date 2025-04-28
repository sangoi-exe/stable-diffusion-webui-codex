import os
import re
import io
from datetime import datetime

try:
    # Attempt to import Tkinter for GUI file dialog
    from tkinter import Tk, filedialog
except ImportError:
    # If Tkinter is not available, set Tk to None
    Tk = None
try:
    # Attempt to import tiktoken for token counting
    import tiktoken
except ImportError:
    # If tiktoken is not available, set it to None
    tiktoken = None

# --- Configuration ---

# Global variable to define the snapshot mode:
# 'ALL': Include all files respecting ignore rules and extensions.
# 'SPECIFIC': Include only files/directories listed in SPECIFIC_TARGETS.
SNAPSHOT_MODE = "SPECIFIC"  # Options: "ALL", "SPECIFIC"

# File extensions to include when SNAPSHOT_MODE is 'ALL'.
include_extensions = {".py", ".js"}

# Directories to completely ignore during traversal (affects both modes).
ignore_dirs = {
    ".git",
    ".vscode",
    "venv",
    "__pycache__", # Added common cache directory
    "node_modules", # Added common JS dependency directory
    "docs",
    "embedding_templates",
    # START: Added potential large/unwanted dirs often found in SD projects
    "outputs",
    "log",
    "models",
    "cache",
    "repositories",
    "tmp",
    "interrupted",
    ".ipynb_checkpoints",
    # END: Added potential large/unwanted dirs
}

# Specific file names to ignore when SNAPSHOT_MODE is 'ALL'.
ignore_files = {
    ".gitignore", # Added common git ignore file
    ".env",       # Added common environment file
    "snapshotLegacy.py", # Example specific file to ignore
    # Note: .git, .vscode, venv are typically handled by ignore_dirs
}

# Set of specific files AND directories to include when SNAPSHOT_MODE is 'SPECIFIC'.
# Paths should be relative to the project root.
# For directories, simply list the directory path.
SPECIFIC_TARGETS = {
    "modules/shared_state.py": True,
    "modules/ui_toprow.py": True,
    "modules/processing.py": True,
    "scripts/xyz_grid.py": True,
    "modules/call_queue.py": True,
    "modules/fifo_lock.py": True,
    "modules/ui.py": True,
    "modules/extras.py": True,
}


# START: Initialize specific file/dir sets (will be populated after project_dir is known)
NORMALIZED_SPECIFIC_FILES = set()
NORMALIZED_SPECIFIC_DIRS = set()
# END: Initialize specific file/dir sets

# Regular expression patterns for files to ignore when SNAPSHOT_MODE is 'ALL'.
ignore_file_patterns = [
    re.compile(r".*\.spec\.(js|py)$"), # Test files
    re.compile(r".*\.min\.(js|css)$"), # Minified assets
    re.compile(r".*\.log$"),           # Log files
    re.compile(r".*\.tmp$"),           # Temporary files
    re.compile(r".*\.swp$"),           # Swap files
    # START: Added potentially large binary/model files
    re.compile(r".*\.(ckpt|safetensors|pt|pth|bin|onnx|pb)$"),
    # END: Added potentially large binary/model files
]

# --- Helper Functions ---

# START: Function to populate normalized specific sets
def initialize_specific_targets(project_root, targets):
    """
    Populates NORMALIZED_SPECIFIC_FILES and NORMALIZED_SPECIFIC_DIRS
    based on the specified targets and the project root directory.
    """
    global NORMALIZED_SPECIFIC_FILES, NORMALIZED_SPECIFIC_DIRS
    NORMALIZED_SPECIFIC_FILES.clear()
    NORMALIZED_SPECIFIC_DIRS.clear()

    print("Normalizando e classificando alvos específicos...")
    checked_targets = 0
    for target in targets:
        normalized_target_rel = os.path.normpath(target)
        full_target_path = os.path.join(project_root, normalized_target_rel)

        # Check if the target exists and classify it
        if os.path.isfile(full_target_path):
            NORMALIZED_SPECIFIC_FILES.add(normalized_target_rel)
            checked_targets += 1
            # print(f"  [Arquivo] {normalized_target_rel}") # Uncomment for debugging
        elif os.path.isdir(full_target_path):
            NORMALIZED_SPECIFIC_DIRS.add(normalized_target_rel)
            checked_targets += 1
            # print(f"  [Diretório] {normalized_target_rel}") # Uncomment for debugging
        else:
            print(f"  [Aviso] Alvo específico não encontrado ou inválido: {target}")

    print(f"Alvos específicos verificados: {checked_targets} (Arquivos: {len(NORMALIZED_SPECIFIC_FILES)}, Diretórios: {len(NORMALIZED_SPECIFIC_DIRS)})")
# END: Function to populate normalized specific sets


def should_include_dir(dir_name, dir_rel_path):
    """
    Checks if a directory should be included in the tree structure or recursed into.
    Uses normalized paths for comparison.
    """
    # Always ignore directories listed in ignore_dirs
    if dir_name in ignore_dirs:
        return False

    # START: Modified logic for SPECIFIC mode
    if SNAPSHOT_MODE == "ALL":
        # In 'ALL' mode, include if not explicitly ignored
        return True
    elif SNAPSHOT_MODE == "SPECIFIC":
        # In 'SPECIFIC' mode, include a directory if:
        # 1. It is explicitly listed.
        # 2. It is an ancestor of an explicitly listed file or directory.
        # 3. It is a descendant of an explicitly listed directory (implicitly handled by recursion).
        normalized_dir_path = os.path.normpath(dir_rel_path)

        # 1. Check if the directory itself is listed
        if normalized_dir_path in NORMALIZED_SPECIFIC_DIRS:
            return True

        # 2. Check if it's an ancestor of any specific file or directory
        # Add os.path.sep to ensure matching directories, not just prefixes
        prefix = normalized_dir_path + os.path.sep
        if any(f.startswith(prefix) for f in NORMALIZED_SPECIFIC_FILES):
            return True
        if any(d.startswith(prefix) for d in NORMALIZED_SPECIFIC_DIRS):
            return True

        # If none of the above, don't include this branch unless it leads to a target
        return False
    # END: Modified logic for SPECIFIC mode

    # Default to not including if mode is unrecognized
    return False


def should_include_file(file_name, file_rel_path):
    """
    Checks if a file should be included based on the current SNAPSHOT_MODE.
    Uses normalized paths for comparison.
    """
    # Normalize the relative path of the file for consistent comparison
    normalized_rel_path = os.path.normpath(file_rel_path)

    if SNAPSHOT_MODE == "ALL":
        # In 'ALL' mode, check against ignore lists, patterns, and extensions.
        # START: Added check for ignore_file_patterns here
        if (
            file_name in ignore_files
            or not file_name.endswith(tuple(include_extensions))
            or any(p.match(normalized_rel_path) for p in ignore_file_patterns) # Check full relative path pattern
            or any(p.match(file_name) for p in ignore_file_patterns) # Check filename pattern
        ):
            return False
        # END: Added check for ignore_file_patterns here
        return True
    elif SNAPSHOT_MODE == "SPECIFIC":
        # START: Modified logic for SPECIFIC mode
        # In 'SPECIFIC' mode, check if:
        # 1. The file itself is explicitly listed.
        # 2. The file resides within an explicitly listed directory.

        # 1. Check if the file itself is listed
        if normalized_rel_path in NORMALIZED_SPECIFIC_FILES:
            return True

        # 2. Check if the file is within any listed directory
        for specific_dir in NORMALIZED_SPECIFIC_DIRS:
            # Ensure we match directories correctly by adding separator
            dir_prefix = specific_dir + os.path.sep
            if normalized_rel_path.startswith(dir_prefix):
                # Additionally, respect ignore_files and ignore_file_patterns even within specific dirs
                if (
                    file_name in ignore_files # Check specific filename ignores
                    or any(p.match(normalized_rel_path) for p in ignore_file_patterns) # Check full path patterns
                    or any(p.match(file_name) for p in ignore_file_patterns) # Check filename patterns
                ):
                    return False # Ignored even if inside specific dir
                return True # Include if not ignored

        # If not explicitly listed or within a listed directory, exclude.
        return False
        # END: Modified logic for SPECIFIC mode

    # Default to not including if mode is unrecognized
    return False

def optimize_content(content, ext):
    """
    Removes comments and excessive blank lines from file content based on extension.
    """
    # Handle JavaScript and CSS: remove block comments and single-line comments
    if ext in {".js", ".css"}:
        # Use non-greedy match for block comments
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL) # Remove /* ... */
        lines = [
            # Strip leading/trailing whitespace before checking for comment start
            l.strip() for l in content.splitlines() if not l.strip().startswith("//") # Remove // ...
        ]
    # Handle Python: remove single-line comments
    elif ext == ".py":
        lines = [
            l.strip() for l in content.splitlines() if not l.strip().startswith("#") # Remove # ...
        ]
    # Handle Handlebars: remove {{! ... }} comments
    elif ext == ".handlebars":
         # Use non-greedy match for comments
        content = re.sub(r"{{!\s*.*?\s*}}", "", content, flags=re.DOTALL) # Remove {{! ... }}
        lines = [l.strip() for l in content.splitlines()]
    # Default: just strip trailing whitespace
    else:
        lines = [l.strip() for l in content.splitlines()] # Use strip() to remove leading/trailing space

    # Remove excessive blank lines (keep at most one consecutive blank line)
    optimized, prev_empty = [], False
    for l in lines:
        # Consider a line empty only if it's truly empty after stripping
        is_empty = (l == "")
        if is_empty:
            # Only add the first blank line after non-blank content
            if not prev_empty:
                optimized.append(l)
            prev_empty = True
        else:
            optimized.append(l)
            prev_empty = False

    # Join lines, ensuring final newline if content exists
    final_content = "\n".join(optimized)
    if final_content and not final_content.endswith('\n'):
         final_content += '\n'
    return final_content

def tree(root, rel, pad, out, print_files):
    """
    Recursively generates the directory tree structure and file content.
    Filters directories and files based on the current SNAPSHOT_MODE and ignore rules.
    Uses normalized paths internally.
    """
    # Construct the full path to the current directory/file
    full = os.path.join(root, rel) if rel else root
    # Normalize the full path for reliability
    normalized_full = os.path.normpath(full)

    try:
        # List items in the current directory using the normalized path
        # Ensure items are sorted for consistent order
        items = sorted(os.listdir(normalized_full))
    except OSError as e: # Catch specific OS errors like permission denied
        out.write(f"{pad}+-- [Erro de acesso: {os.path.basename(normalized_full)} - {e}]\n")
        return # Stop recursion for this branch on error

    dirs, files = [], []
    for item in items:
        # Full path of the item for type checking (isdir/isfile)
        item_full_path = os.path.join(normalized_full, item)
        # Relative path of the item for inclusion logic and display
        item_rel_path = os.path.join(rel, item) if rel else item
        # Normalize the relative path for consistent checks
        normalized_item_rel_path = os.path.normpath(item_rel_path)

        # Check if it's a directory
        if os.path.isdir(item_full_path):
            # Use helper function to decide inclusion based on mode and ignores
            # Pass normalized relative path to helper
            if should_include_dir(item, normalized_item_rel_path):
                dirs.append(item)
        # Check if it's a file
        elif os.path.isfile(item_full_path):
            # Use helper function to decide inclusion based on mode and ignores
            # Pass normalized relative path to helper
            if should_include_file(item, normalized_item_rel_path):
                files.append(item)

    # Process included files first (already sorted from listdir)
    for f in files:
        # Construct the relative path for display (use the non-normalized original if needed for display)
        display_rel_path = os.path.join(rel, f) if rel else f
        # Write the file entry, normalizing the path for consistent display
        out.write(f"{pad}+-- {os.path.normpath(display_rel_path)}\n")
        # If requested, print the optimized content of the file
        if print_files:
            try:
                # Construct the full path to read the file
                file_to_read_path = os.path.join(normalized_full, f)
                # START: Read file content safely
                try:
                    with open(
                        file_to_read_path, "r", encoding="utf-8", errors="strict" # Try strict first
                    ) as fc:
                        content = fc.read()
                except UnicodeDecodeError:
                    # If strict fails, try replacing errors
                    print(f"    Aviso: Falha na decodificação UTF-8 estrita para {display_rel_path}. Tentando com 'replace'.")
                    with open(
                        file_to_read_path, "r", encoding="utf-8", errors="replace"
                    ) as fc:
                        content = fc.read()
                # END: Read file content safely

                # Get file extension for optimization logic
                ext = os.path.splitext(f)[1].lower()
                optimized_c = optimize_content(content, ext)

                # Only write content if it's not empty after optimization
                if optimized_c.strip():
                    # Write the optimized content within a code block
                    # Ensure correct indentation for the code block itself
                    out.write(
                        f"{pad}    ``` {os.path.basename(f)} \n" # Use basename for clarity in block
                        f"{optimized_c}" # Content already has trailing newline if needed
                        f"{pad}    ```\n\n"
                    )
                else:
                    # Indicate if file is empty after optimization
                     out.write(f"{pad}    [Arquivo vazio após otimização]\n\n")

            except Exception as e:
                # Report errors reading specific files
                out.write(f"{pad}    [Erro ao ler/processar {f}: {e}]\n\n")

    # Process included directories next (already sorted)
    for d in dirs:
        # Construct the relative path for the subdirectory
        new_rel = os.path.join(rel, d) if rel else d
        # Write the directory entry, normalizing path and adding separator for clarity
        out.write(f"{pad}+-- {os.path.normpath(new_rel)}{os.path.sep}\n")
        # Recurse into the subdirectory with increased padding
        tree(root, new_rel, pad + "    ", out, print_files)

def count_tokens(text):
    """
    Counts tokens in the given text using tiktoken if available,
    otherwise falls back to a simple word count.
    """
    if tiktoken:
        try:
            # Try encoding for a common model like gpt-4 or gpt-3.5-turbo
            # cl100k_base is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"Aviso: Falha ao obter codificação tiktoken 'cl100k_base' ({e}). Contagem de tokens pode ser imprecisa.")
            # Basic fallback: count space-separated words
            return len(text.split())
        # Encode the text to get token IDs
        return len(enc.encode(text, disallowed_special=())) # Allow special tokens for more accurate count
    else:
        # Basic fallback: count space-separated words
        return len(text.split())

# --- Main Execution ---

if __name__ == "__main__":
    # Select project directory using GUI if Tkinter is available, else use CWD
    if Tk:
        root_tk = Tk()
        root_tk.withdraw() # Hide the main Tk window
        project_dir_selected = filedialog.askdirectory(
            title="Selecione a pasta do projeto"
        )
        root_tk.destroy() # Destroy the Tk instance after use
        # Use selected directory or fallback to current working directory if canceled
        if not project_dir_selected:
             print("Nenhum diretório selecionado. Saindo.")
             exit()
        project_dir = project_dir_selected
    else:
        # Use current working directory if Tkinter is not available
        project_dir = os.getcwd()
        print("Tkinter não encontrado ou falhou ao inicializar. Usando o diretório de trabalho atual:", project_dir)
        # Optionally prompt user for path in console
        # response = input(f"Pressione Enter para usar {project_dir} ou digite um novo caminho: ")
        # if response.strip():
        #     project_dir = response.strip()

    # Normalize the project directory path for consistency
    project_dir = os.path.normpath(project_dir)
    if not os.path.isdir(project_dir):
        print(f"Erro: O diretório do projeto especificado não existe ou não é um diretório: {project_dir}")
        exit()

    print(f"Diretório do projeto selecionado: {project_dir}")
    print(f"Modo de Snapshot: {SNAPSHOT_MODE}")

    # START: Initialize specific targets after getting project_dir
    if SNAPSHOT_MODE == "SPECIFIC":
        initialize_specific_targets(project_dir, SPECIFIC_TARGETS)
        if not NORMALIZED_SPECIFIC_FILES and not NORMALIZED_SPECIFIC_DIRS:
            print("Aviso: Modo 'SPECIFIC' selecionado, mas nenhum arquivo ou diretório válido foi encontrado nos alvos especificados.")
            # Decide whether to exit or continue with an empty snapshot
            # exit() # Option 1: Exit
            print("Continuando para gerar um snapshot potencialmente vazio...") # Option 2: Continue
    # END: Initialize specific targets

    # Generate timestamp for the snapshot file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Generate Directory Structure (Simple Tree) ---
    print("Gerando estrutura de diretórios (árvore simples)...")
    # Create an in-memory buffer for the simple tree
    simple_tree_buffer = io.StringIO()
    # Call tree function without printing file contents (print_files=False)
    tree(project_dir, "", "", simple_tree_buffer, False)
    # Get the generated tree string from the buffer
    dir_tree_str = simple_tree_buffer.getvalue()
    # Add a fallback message if the tree is empty
    if not dir_tree_str.strip():
        dir_tree_str = "[Nenhuma estrutura de diretório para exibir com base nos filtros atuais]\n"


    # --- Calculate Tokens per Root Folder (Only for 'ALL' Mode) ---
    folder_tokens = {}
    tokens_section = "N/A (Modo Específico)\n"  # Default message for SPECIFIC mode

    if SNAPSHOT_MODE == "ALL":
        tokens_section = ""  # Reset for ALL mode
        print("Calculando tokens por pasta raiz (Modo ALL)...")
        try:
            # Iterate through items directly in the project root
            for item in sorted(os.listdir(project_dir)):
                path = os.path.join(project_dir, item)
                # Relative path for root items is just the item name
                item_rel_path = item
                # Check if it's a directory and should be included (using normalized path)
                if os.path.isdir(path) and should_include_dir(item, os.path.normpath(item_rel_path)):
                    # Create a buffer for the content of this specific folder
                    buf = io.StringIO()
                    # Generate the tree *with file content* for this folder only
                    tree(project_dir, item, "", buf, True)
                    folder_content = buf.getvalue()
                    # Count tokens only if there's actual content
                    if folder_content.strip():
                        folder_tokens[item] = count_tokens(folder_content)
        except Exception as e:
             tokens_section = f"Erro ao calcular tokens por pasta: {e}\n"

        # Format the tokens section if calculations were successful
        if not tokens_section: # Check if error message wasn't set
            if folder_tokens:
                for folder, token_count in sorted(folder_tokens.items()):
                    tokens_section += f"{folder}: {token_count} tokens\n"
            else:
                tokens_section = "Nenhuma pasta raiz aplicável encontrada para contagem de tokens.\n"
            # Ensure a trailing newline if content exists
            if tokens_section and not tokens_section.endswith('\n'):
                tokens_section += '\n'

    # --- Generate Detailed Snapshot (Tree + Content) ---
    print("Gerando snapshot detalhado (conteúdo)...")
    # Create an in-memory buffer for the detailed snapshot
    detailed_buffer = io.StringIO()
    # Call tree function *with* printing file contents (print_files=True)
    tree(project_dir, "", "", detailed_buffer, True)
    # Get the generated detailed snapshot string
    detailed_snapshot = detailed_buffer.getvalue()
     # Add a fallback message if the detailed content is empty
    if not detailed_snapshot.strip():
        detailed_snapshot = "[Nenhum conteúdo de arquivo para exibir com base nos filtros atuais]\n"


    # --- Assemble Final Snapshot ---
    print("Montando o arquivo final...")
    # Define the header template, including the mode and token section
    # Ensure dir_tree_str and tokens_section end with a newline for formatting
    dir_tree_str_formatted = dir_tree_str if dir_tree_str.endswith('\n') else dir_tree_str + '\n'
    tokens_section_formatted = tokens_section if tokens_section.endswith('\n') else tokens_section + '\n'

    header_template = f"""# Snapshot do Projeto (Modo: {SNAPSHOT_MODE})
Timestamp: {timestamp}
Diretório Raiz: {project_dir}
Tokens Totais: {{TOTAL_TOKENS}}

## Estrutura do Projeto:
```
{dir_tree_str_formatted}```

## Tokens por Pasta Raiz (Apenas Modo ALL):
```
{tokens_section_formatted}```

## Conteúdo do Projeto:
"""

    # Combine header template and detailed content
    # Placeholder {TOTAL_TOKENS} will be replaced after counting
    # Use temporary content for token counting to avoid double concatenation issues
    content_for_token_count = header_template.replace("{TOTAL_TOKENS}", "0") + detailed_snapshot

    # Calculate total tokens for the entire snapshot content
    print("Calculando tokens totais...")
    total_tokens = count_tokens(content_for_token_count)

    # Replace the placeholder with the actual total token count in the final string
    final_snapshot_content = header_template.replace(
        "{TOTAL_TOKENS}", str(total_tokens)
    ) + detailed_snapshot

    # --- Write to File ---
    # Get the base name of the project folder for the output filename
    folder_name = os.path.basename(project_dir)
    # Construct the output filename including folder name, mode, and timestamp
    output_file = f"snapshot_{folder_name}_{SNAPSHOT_MODE}_{timestamp}.txt"
    # Save in the *parent* directory of the selected project_dir if possible, otherwise CWD
    output_dir = os.path.dirname(project_dir) or os.getcwd()
    output_path = os.path.join(output_dir, output_file)

    try:
        print(f"Salvando snapshot em: {output_path}")
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(final_snapshot_content)
        print("Snapshot salvo com sucesso!")
        print(f"Tokens totais estimados: {total_tokens}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo de snapshot: {e}")
