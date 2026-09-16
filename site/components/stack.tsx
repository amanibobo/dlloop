import * as si from "simple-icons";

/* The tools and languages behind the project, as a strip of monochrome icons that take their
   brand colour on hover, with a small tooltip. Icons are inlined from simple-icons at build time;
   packages without a logo get a monogram badge in the same style. */

type Item = { name: string; path?: string; hex?: string; mono?: string; href: string };

const brand = (icon: { title: string; path: string; hex: string }, href: string): Item => ({ name: icon.title, path: icon.path, hex: `#${icon.hex}`, href });

const STACK: Item[] = [
  brand(si.siPython, "https://www.python.org/"),
  brand(si.siPydantic, "https://ai.pydantic.dev/"),
  { name: "Lenstronomy", mono: "Ln", hex: "#2a78d6", href: "https://github.com/lenstronomy/lenstronomy" },
  { name: "PyAutoLens", mono: "AL", hex: "#1baf7a", href: "https://github.com/Jammy2211/PyAutoLens" },
  brand(si.siNumpy, "https://numpy.org/"),
  brand(si.siScipy, "https://scipy.org/"),
  brand(si.siPytorch, "https://pytorch.org/"),
  brand(si.siModal, "https://modal.com/"),
  brand(si.siUv, "https://docs.astral.sh/uv/"),
  { name: "marimo", mono: "mo", hex: "#0ca30c", href: "https://marimo.io/" },
  brand(si.siTypescript, "https://www.typescriptlang.org/"),
  brand(si.siReact, "https://react.dev/"),
  brand(si.siNextdotjs, "https://nextjs.org/"),
  brand(si.siTailwindcss, "https://tailwindcss.com/"),
  brand(si.siVercel, "https://vercel.com/"),
  brand(si.siGithub, "https://github.com/amanibobo/lensgrav"),
];

export function Stack() {
  return (
    <div>
      <p className="kicker">Built with</p>
      <ul className="mt-3 flex flex-wrap gap-2">
        {STACK.map((t) => (
          <li key={t.name} className="group relative">
            <a
              href={t.href}
              target="_blank"
              rel="noreferrer"
              aria-label={t.name}
              className="flex h-11 w-11 items-center justify-center rounded-lg border border-line bg-white text-ink-2 transition-colors hover:border-[var(--brand)] hover:text-[var(--brand)] focus-visible:border-[var(--brand)] focus-visible:text-[var(--brand)] focus-visible:outline-none"
              style={{ ["--brand" as string]: t.hex === "#000000" || t.hex === "#181717" ? "#111111" : t.hex }}
            >
              {t.path ? (
                <svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor" aria-hidden>
                  <path d={t.path} />
                </svg>
              ) : (
                <span className="font-mono text-[0.8rem] font-semibold tracking-tight" aria-hidden>
                  {t.mono}
                </span>
              )}
            </a>
            <span
              role="tooltip"
              className="pointer-events-none absolute left-1/2 top-full z-10 mt-2 -translate-x-1/2 whitespace-nowrap rounded-md bg-ink px-2 py-1 text-[0.72rem] text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
            >
              {t.name}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
