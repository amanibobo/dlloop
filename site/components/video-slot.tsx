"use client";

import { useEffect, useRef, useState } from "react";

export function VideoSlot({ file, title, caption }: { file: string; title: string; caption: string }) {
  const [state, setState] = useState<"checking" | "ready" | "missing">("checking");
  const ref = useRef<HTMLVideoElement>(null);
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

  // play while on screen, pause when scrolled away (autoplay needs muted, and these are silent)
  useEffect(() => {
    const el = ref.current;
    if (state !== "ready" || !el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) el.play().catch(() => {});
        else el.pause();
      },
      { threshold: 0.4 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [state]);

  return (
    <figure className="my-10">
      <div className={`relative overflow-hidden rounded-xl border border-line bg-surface ${state === "ready" ? "" : "aspect-video"}`}>
        {state === "ready" ? (
          <video ref={ref} className="block h-auto w-full" controls muted loop playsInline preload="metadata" src={src} onError={() => setState("missing")} />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-1.5 p-6 text-center">
            <span className="kicker">{state === "checking" ? "Loading" : "Recording coming soon"}</span>
            {state === "missing" ? (
              <span className="text-sm text-muted">
                <code className="font-mono">{file}</code>
              </span>
            ) : null}
          </div>
        )}
      </div>
      <figcaption className="mt-4 text-center">
        <span className="block text-[0.95rem] text-ink">{title}</span>
        {caption ? <span className="mt-1 block text-[0.9rem] leading-relaxed text-ink-2">{caption}</span> : null}
      </figcaption>
    </figure>
  );
}
