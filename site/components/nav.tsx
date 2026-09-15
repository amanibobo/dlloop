"use client";

import { useEffect, useState } from "react";
import { nav, REPO } from "@/content/site";

export function Nav() {
  const [active, setActive] = useState<string>("");

  useEffect(() => {
    const sections = nav.map((n) => document.getElementById(n.id)).filter((el): el is HTMLElement => el !== null);
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-40% 0px -55% 0px" },
    );
    sections.forEach((s) => observer.observe(s));
    return () => observer.disconnect();
  }, []);

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center gap-4 px-5 py-3 sm:px-8">
        <a href="#top" className="font-semibold tracking-tight text-ink">
          LensCraft
        </a>
        <nav className="hidden flex-1 gap-1 overflow-x-auto md:flex" aria-label="Sections">
          {nav.map((n) => (
            <a
              key={n.id}
              href={`#${n.id}`}
              className={`rounded-md px-2.5 py-1 text-sm transition-colors ${active === n.id ? "bg-surface-2 text-ink" : "text-ink-2 hover:text-ink"}`}
            >
              {n.label}
            </a>
          ))}
        </nav>
        <a href={REPO} className="ml-auto text-sm text-ink-2 underline underline-offset-4 hover:text-ink" target="_blank" rel="noreferrer">
          GitHub ↗
        </a>
      </div>
    </header>
  );
}
