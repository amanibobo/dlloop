"use client";

import { useEffect, useState } from "react";

export function VideoSlot({ file, title, caption }: { file: string; title: string; caption: string }) {
  const [state, setState] = useState<"checking" | "ready" | "missing">("checking");
  const src = `/videos/${file}`;

  useEffect(() => {
    let cancelled = false;
    fetch(src, { method: "HEAD" })
      .then((res) => {
        const type = res.headers.get("content-type") ?? "";
        if (!cancelled) setState(res.ok && type.startsWith("video/") ? "ready" : "missing");
      })
      .catch(() => {
        if (!cancelled) setState("missing");
      });
    return () => {
      cancelled = true;
    };
  }, [src]);

  return (
    <figure className="flex flex-col">
      <div className="relative aspect-video overflow-hidden rounded-lg border border-line bg-surface-2">
        {state === "ready" ? (
          <video className="h-full w-full bg-black" controls preload="metadata" playsInline src={src} onError={() => setState("missing")} />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
            <span className="font-mono text-xs uppercase tracking-widest text-muted">{state === "checking" ? "loading" : "recording pending"}</span>
            {state === "missing" ? (
              <span className="text-sm text-ink-2">
                Drop <code className="font-mono">{file}</code> into <code className="font-mono">public/videos/</code>
              </span>
            ) : null}
          </div>
        )}
      </div>
      <figcaption className="mt-3">
        <span className="block font-medium text-ink">{title}</span>
        <span className="mt-1 block text-sm leading-relaxed text-ink-2">{caption}</span>
      </figcaption>
    </figure>
  );
}
