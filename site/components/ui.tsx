import type { ReactNode } from "react";

/* Column: 780px, centred, generous vertical rhythm. Sections are typographic, not boxed. */

export function Section({ id, title, children }: { id: string; kicker?: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-16 py-12 sm:py-16">
      <div className="mx-auto w-full max-w-195 px-6">
        <h2 className="text-[1.55rem] font-semibold leading-[1.22] tracking-[-0.01em] text-ink sm:text-[1.75rem]">{title}</h2>
        <div className="mt-5">{children}</div>
      </div>
    </section>
  );
}

export function Sub({ id, children }: { id?: string; children: ReactNode }) {
  return (
    <h3 id={id} className="mt-12 scroll-mt-16 text-[1.15rem] font-semibold tracking-[-0.01em] text-ink">
      {children}
    </h3>
  );
}

export function Prose({ children }: { children: ReactNode }) {
  return <div className="prose">{children}</div>;
}

export function Figure({ src, alt, caption }: { src: string; alt: string; caption?: string }) {
  return (
    <figure className="my-8">
      <div className="overflow-hidden rounded-xl border border-line bg-white">
        {/* eslint-disable-next-line @next/next/no-img-element -- static PNGs from the report generator */}
        <img src={src} alt={alt} className="h-auto w-full" />
      </div>
      {caption ? <figcaption className="mt-3 text-center text-[0.85rem] leading-relaxed text-ink-2">{caption}</figcaption> : null}
    </figure>
  );
}

export function Table({ head, rows, className = "" }: { head: string[]; rows: (string | ReactNode)[][]; className?: string }) {
  return (
    <div className={`my-6 overflow-x-auto ${className}`}>
      <table className="w-full text-[0.86rem]">
        <thead>
          <tr className="border-b border-line text-left text-muted">
            {head.map((h) => (
              <th key={h} className="py-2 pr-5 font-normal">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line">
              {r.map((c, j) => (
                <td key={j} className={`py-2.5 pr-5 align-top ${j === 0 ? "text-ink" : "text-ink-2"}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ClassDot({ cls }: { cls: "none" | "subhalo" | "vortex" }) {
  const color = { none: "bg-none", subhalo: "bg-subhalo", vortex: "bg-vortex" }[cls];
  return (
    <span className="inline-flex items-center gap-2">
      <span className={`inline-block h-2 w-2 rounded-full ${color}`} aria-hidden />
      {cls}
    </span>
  );
}

export function Code({ children }: { children: string }) {
  return (
    <pre className="my-2.5 overflow-x-auto rounded-lg bg-surface px-4 py-2.5 font-mono text-[0.78rem] leading-relaxed text-ink">
      <code>{children}</code>
    </pre>
  );
}

export function Meta({ items }: { items: { label: string; lines: string[] }[] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4">
      {items.map((it) => (
        <div key={it.label}>
          <dt className="kicker">{it.label}</dt>
          <dd className="mt-1.5 text-[0.95rem] leading-relaxed text-ink-2">
            {it.lines.map((l) => (
              <span key={l} className="block">
                {l}
              </span>
            ))}
          </dd>
        </div>
      ))}
    </dl>
  );
}
