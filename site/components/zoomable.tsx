"use client";

import { useCallback, useEffect, useState } from "react";

/* An image that shows a zoom-in cursor on hover and opens enlarged over a dimmed backdrop on
   click. Clicking anywhere, or pressing Escape, closes it. */
export function Zoomable({ src, alt, className = "" }: { src: string; alt: string; className?: string }) {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, close]);

  return (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element -- static image */}
      <img src={src} alt={alt} className={`cursor-zoom-in ${className}`} onClick={() => setOpen(true)} />
      {open ? (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={alt}
          onClick={close}
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/70 p-6 backdrop-blur-sm sm:p-12"
        >
          {/* eslint-disable-next-line @next/next/no-img-element -- static image */}
          <img src={src} alt={alt} className="max-h-full max-w-full rounded-xl bg-white object-contain shadow-2xl" />
        </div>
      ) : null}
    </>
  );
}
