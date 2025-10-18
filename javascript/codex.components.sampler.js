(function () {
  const NS = (window.Codex = window.Codex || {});
  const C = (NS.Components = NS.Components || {});

  function find(id) {
    try {
      return gradioApp().querySelector(`#${id} select`) || gradioApp().querySelector(`#${id}`);
    } catch (e) {
      return null;
    }
  }

  C.Sampler = {
    setChoices(ids, choices) {
      const [samplerId, schedulerId] = ids;
      const samplerEl = find(samplerId);
      const schedulerEl = find(schedulerId);
      if (samplerEl && Array.isArray(choices.samplers)) {
        samplerEl.innerHTML = '';
        choices.samplers.forEach((label) => {
          const opt = document.createElement('option');
          opt.value = label; opt.textContent = label; samplerEl.appendChild(opt);
        });
      }
      if (schedulerEl && Array.isArray(choices.schedulers)) {
        schedulerEl.innerHTML = '';
        choices.schedulers.forEach((label) => {
          const opt = document.createElement('option');
          opt.value = label; opt.textContent = label; schedulerEl.appendChild(opt);
        });
      }
    }
  };
})();

