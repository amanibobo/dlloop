"use client";

import { useEffect, useState } from "react";

/* An image slot that renders `/media/<file>` when it exists and a quiet placeholder otherwise.
   Used for hand-drawn (tldraw) diagrams and screenshots the author adds after the fact. */
export function MediaSlot({ file, alt, caption, kind = "sketch" }: { file: string; alt: string; caption?: string; kind?: "sketch" | "screenshot" }) {
  const [state, setState] = useState<"checking" | "ready" | "missing">("checking");
  const src = `/media/${file}`;

  useEffect(() => {
    let cancelled = false;
    fetch(src, { method: "HEAD" })
      .then((res) => {
        const type = res.headers.get("content-type") ?? "";
        if (!cancelled) setState(res.ok && type.startsWith("image/") ? "ready" : "missing");
      })
      .catch(() => {
        if (!cancelled) setState("missing");
      });
    return () => {
      cancelled = true;
    };
  }, [src]);

  if (state === "checking") return <div className="my-8 h-40" aria-hidden />;

  if (state === "missing") {
    return (
      <figure className="my-8">
        <div className="flex h-40 flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-line text-center">
          <span className="kicker">{kind === "sketch" ? "Hand-drawn diagram goes here" : "Screenshot goes here"}</span>
          <span className="font-mono text-[0.78rem] text-muted">public/media/{file}</span>
        </div>
        {caption ? <figcaption className="mt-3 text-center text-[0.85rem] leading-relaxed text-muted">{caption}</figcaption> : null}
      </figure>
    );
  }

  return (
    <figure className="my-8">
      <div className={kind === "screenshot" ? "overflow-hidden rounded-xl border border-line bg-white" : ""}>
        {/* eslint-disable-next-line @next/next/no-img-element -- author-supplied static image */}
        <img src={src} alt={alt} className="h-auto w-full" />
      </div>
      {caption ? <figcaption className="mt-3 text-center text-[0.85rem] leading-relaxed text-ink-2">{caption}</figcaption> : null}
    </figure>
  );
}
