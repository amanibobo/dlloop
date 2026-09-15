import type { ReactNode } from "react";

export function Section({ id, kicker, title, lede, children }: { id: string; kicker: string; title: string; lede?: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 border-t border-line py-16 sm:py-24">
      <div className="mx-auto w-full max-w-5xl px-5 sm:px-8">
        <p className="font-mono text-xs uppercase tracking-[0.18em] text-muted">{kicker}</p>
        <h2 className="mt-2 text-3xl font-semibold tracking-tight text-ink sm:text-4xl">{title}</h2>
        {lede ? <p className="prose mt-4 text-lg text-ink-2">{lede}</p> : null}
        <div className="mt-10">{children}</div>
      </div>
    </section>
  );
}

export function Prose({ children }: { children: ReactNode }) {
  return <div className="prose">{children}</div>;
}

export function Figure({ src, alt, caption, wide = false }: { src: string; alt: string; caption?: string; wide?: boolean }) {
  return (
    <figure className={`my-8 ${wide ? "" : "max-w-3xl"}`}>
      <div className="overflow-hidden rounded-lg border border-line bg-surface p-2">
        {/* eslint-disable-next-line @next/next/no-img-element -- static PNGs from the report generator */}
        <img src={src} alt={alt} className="h-auto w-full rounded" />
      </div>
      {caption ? <figcaption className="mt-3 text-sm leading-relaxed text-ink-2">{caption}</figcaption> : null}
    </figure>
  );
}

export function Details({ summary, children }: { summary: string; children: ReactNode }) {
  return (
    <details className="group my-6 rounded-lg border border-line bg-surface px-5 py-4">
      <summary className="font-medium text-ink">{summary}</summary>
      <div className="prose mt-4 text-[0.97rem] text-ink-2">{children}</div>
    </details>
  );
}

export function Table({ head, rows, className = "" }: { head: string[]; rows: (string | ReactNode)[][]; className?: string }) {
  return (
    <div className={`my-6 overflow-x-auto rounded-lg border border-line bg-surface ${className}`}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wider text-muted">
            {head.map((h) => (
              <th key={h} className="px-4 py-3 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line last:border-0">
              {r.map((c, j) => (
                <td key={j} className={`px-4 py-3 align-top ${j === 0 ? "font-medium text-ink" : "text-ink-2"}`}>
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
      <span className={`inline-block h-2.5 w-2.5 rounded-full ${color}`} aria-hidden />
      {cls}
    </span>
  );
}

export function Code({ children }: { children: string }) {
  return (
    <pre className="my-4 overflow-x-auto rounded-lg border border-line bg-surface-2 px-4 py-3 font-mono text-[0.85rem] leading-relaxed text-ink">
      <code>{children}</code>
    </pre>
  );
}
