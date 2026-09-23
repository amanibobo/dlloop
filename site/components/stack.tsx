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
    <p className="text-[0.95rem] leading-relaxed text-ink-2">
      Built with:{" "}
      {STACK.map((t, i) => (
        <span key={t.name}>
          <a href={t.href} target="_blank" rel="noreferrer" className="underline decoration-line underline-offset-4 transition-colors hover:text-ink hover:decoration-ink">
            {t.name}
          </a>
          {i < STACK.length - 1 ? ", " : ""}
        </span>
      ))}
    </p>
  );
}
