**Scope**
- Zero compat legacy. Sem adapters, sem rotas antigas, sem fallback.

**API Changes (propostos)**
- Introdução dos tipos de request/resposta (novos) e corte imediato para `orchestrator.run()` com `task` e `engine_key`.

**Adapters**
- Não haverá adapters. As rotas antigas deixam de ser chamadas; os handlers passam a construir Requests e chamar o orquestrador diretamente.

**Compat Matrix**
- `sd15`/`sdxl`/`flux`: chaves oficiais de engine. Aliases legados não serão anunciados; manteremos apenas o necessário para funcionamento interno.

**Removal Timeline**
- Corte imediato (N): trocar handlers para o novo orquestrador e remover referências às rotas antigas.
