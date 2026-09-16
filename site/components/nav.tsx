"use client";

import { useEffect, useState } from "react";
import { nav } from "@/content/site";

/* A fixed left rail of section ticks; the active one darkens as you scroll. No top bar. */
export function Rail() {
  const [active, setActive] = useState<string>("");

  useEffect(() => {
    const sections = nav.map((n) => document.getElementById(n.id)).filter((el): el is HTMLElement => el !== null);
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-35% 0px -55% 0px" },
    );
    sections.forEach((s) => observer.observe(s));
    return () => observer.disconnect();
  }, []);

  return (
    <nav aria-label="Sections" className="fixed left-8 top-1/2 z-10 hidden -translate-y-1/2 flex-col gap-2 lg:flex">
      {nav.map((n) => (
        <a key={n.id} href={`#${n.id}`} title={n.label} className="group flex h-3 items-center">
          <span className={`block h-px transition-all ${active === n.id ? "w-6 bg-ink" : "w-5 bg-line group-hover:bg-muted"}`} />
        </a>
      ))}
    </nav>
  );
}
