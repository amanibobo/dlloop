/* An embedded marimo app (WASM export under /marimo/<name>/, or a server URL). Lazy so the
   page stays light; the first load of a WASM notebook takes a few seconds. */
export function Embed({ src, title, height = 560, caption }: { src: string; title: string; height?: number; caption?: string }) {
  if (!src) {
    return (
      <figure className="my-8">
        <div className="flex h-40 flex-col items-center justify-center rounded-xl border border-dashed border-line text-center">
          <span className="kicker">Interactive widget coming soon</span>
          <span className="mt-1 font-mono text-[0.78rem] text-muted">{title}</span>
        </div>
      </figure>
    );
  }
  return (
    <figure className="my-8">
      <div className="overflow-hidden rounded-xl border border-line bg-white">
        <iframe src={src} title={title} loading="lazy" className="block w-full" style={{ height }} allow="clipboard-write" />
      </div>
      {caption ? <figcaption className="mt-3 text-center text-[0.85rem] leading-relaxed text-ink-2">{caption}</figcaption> : null}
    </figure>
  );
}
