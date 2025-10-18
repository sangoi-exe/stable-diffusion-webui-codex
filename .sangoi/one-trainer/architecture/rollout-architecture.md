**PR Order**
1) Core types + registry + orchestrator.
2) Engines SD15/SDXL/FLUX funcionais (txt2img/img2img mínimos) com progress + métricas.
3) UI: trocar handlers para orquestrador (corte imediato, sem adapters).
4) Engines de vídeo (txt2vid/img2vid) + novas abas.
5) Presets de vídeo + integração no UI e scripts.
6) Limpeza de deprecações (N+2).

**Validation per PR (manual only)**
- Executar roteiros manuais documentados em cada PR (prints/logs). Nada de suites `pytest` ou smoke automatizados.
- Capturar logs completos com task, engine, modelo, preset/applied args, tempo e métricas de VRAM.
- Benchmarks/VRAM via execução manual de scripts/tarefas (documentar comandos/resultados).

**Risk Mitigation**
- Sem flags/fallback: corte direto. Em caso de falha, reverter commit via Git (sem camadas de compatibilidade).
- Erros devem manter stack trace e contexto completo (sem mensagens genéricas); logs precisam ser verbosos para testes manuais.
