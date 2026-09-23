/* The tools and languages behind the project, as a row of plain text links. */

const STACK: { name: string; href: string }[] = [
  { name: "Python", href: "https://www.python.org/" },
  { name: "Pydantic AI", href: "https://ai.pydantic.dev/" },
  { name: "Lenstronomy", href: "https://github.com/lenstronomy/lenstronomy" },
  { name: "PyAutoLens", href: "https://github.com/Jammy2211/PyAutoLens" },
  { name: "NumPy", href: "https://numpy.org/" },
  { name: "SciPy", href: "https://scipy.org/" },
  { name: "PyTorch", href: "https://pytorch.org/" },
  { name: "Modal", href: "https://modal.com/" },
  { name: "marimo", href: "https://marimo.io/" },
  { name: "TypeScript", href: "https://www.typescriptlang.org/" },
];

export function Stack() {
  return (
    <div>
      <p className="kicker">Built with</p>
      <ul className="mt-2 flex flex-wrap gap-x-5 gap-y-1.5 text-[0.95rem]">
        {STACK.map((t) => (
          <li key={t.name}>
            <a href={t.href} target="_blank" rel="noreferrer" className="text-ink-2 underline decoration-line underline-offset-4 transition-colors hover:text-ink hover:decoration-ink">
              {t.name}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
