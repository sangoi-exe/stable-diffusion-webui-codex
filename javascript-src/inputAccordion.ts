function inputAccordionChecked(id: string, checked: boolean): void {
  const app = gradioApp() as Document;
  const accordion = (app.getElementById(id) as HTMLElement | null);
  if (!accordion) return;

  // @ts-expect-error injected by setupAccordion
  if (accordion.visibleCheckbox) {
    // @ts-expect-error injected by setupAccordion
    accordion.visibleCheckbox.checked = checked;
    // @ts-expect-error injected by setupAccordion
    accordion.onVisibleCheckboxChange();
  }
}

function setupAccordion(accordion: HTMLElement): void {
  const app = gradioApp() as Document;

  const labelWrap = accordion.querySelector('.label-wrap') as HTMLElement | null;
  const gradioCheckbox = app.querySelector(`#${accordion.id}-checkbox input`) as HTMLInputElement | null;
  const extra = app.querySelector(`#${accordion.id}-extra`) as HTMLElement | null;
  const span = labelWrap?.querySelector('span') as HTMLSpanElement | null;
  if (!labelWrap || !span) return;

  const isOpen = () => labelWrap.classList.contains('open');

  const observerAccordionOpen = new MutationObserver(() => {
    accordion.classList.toggle('input-accordion-open', isOpen());
    // @ts-expect-error injected below
    if (accordion.visibleCheckbox && accordion.onVisibleCheckboxChange) {
      // keep checkbox state in sync with open/close
      // @ts-expect-error injected below
      accordion.visibleCheckbox.checked = isOpen();
      // @ts-expect-error injected below
      accordion.onVisibleCheckboxChange();
    }
  });
  observerAccordionOpen.observe(labelWrap, { attributes: true, attributeFilter: ['class'] });

  if (extra) {
    labelWrap.insertBefore(extra, labelWrap.lastElementChild);
  }

  // @ts-expect-error dynamic property for legacy code
  accordion.onChecked = (checked: boolean) => {
    if (isOpen() !== checked) labelWrap.click();
  };

  const visibleCheckbox = document.createElement('input');
  visibleCheckbox.type = 'checkbox';
  visibleCheckbox.checked = isOpen();
  visibleCheckbox.id = `${accordion.id}-visible-checkbox`;
  visibleCheckbox.className = `${gradioCheckbox?.className ?? ''} input-accordion-checkbox`;
  span.insertBefore(visibleCheckbox, span.firstChild);

  // @ts-expect-error attach for legacy code
  accordion.visibleCheckbox = visibleCheckbox;
  // @ts-expect-error attach for legacy code
  accordion.onVisibleCheckboxChange = () => {
    if (isOpen() !== visibleCheckbox.checked) labelWrap.click();
    if (gradioCheckbox) {
      gradioCheckbox.checked = visibleCheckbox.checked;
      updateInput(gradioCheckbox);
    }
  };

  visibleCheckbox.addEventListener('click', (ev) => {
    ev.stopPropagation();
  });
  // @ts-expect-error property exists on input
  visibleCheckbox.addEventListener('input', accordion.onVisibleCheckboxChange);
}

onUiLoaded(() => {
  const app = gradioApp() as Document;
  app.querySelectorAll<HTMLElement>('.input-accordion').forEach(setupAccordion);
});

// Expose legacy function
// @ts-expect-error global attach for legacy consumers
window.inputAccordionChecked = inputAccordionChecked;

