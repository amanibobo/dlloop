"use client";

import { useEffect, useState } from "react";
import { nav, REPO } from "@/content/site";

const top = [
  { href: "#overview", label: "Overview" },
  { href: "#build", label: "Process" },
  { href: "#results", label: "Results" },
  { href: "#demos", label: "Demos" },
  { href: REPO, label: "Code" },
];

/* Top bar like the reference (name left, quiet links right) plus a fixed left rail of section
   ticks whose active tick darkens as you scroll. */
export function Nav() {
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
    <>
      <header className="sticky top-0 z-20 border-b border-line bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-295 items-center px-6 py-5">
          <a href="#top" className="text-[1.05rem] font-medium text-ink">
            LensCraft
          </a>
          <nav className="ml-auto flex gap-7" aria-label="Sections">
            {top.map((t) => (
              <a
                key={t.label}
                href={t.href}
                className="text-[0.95rem] text-ink-2 transition-colors hover:text-ink"
                {...(t.href.startsWith("http") ? { target: "_blank", rel: "noreferrer" } : {})}
              >
                {t.label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <nav aria-label="Progress" className="fixed left-8 top-1/2 z-10 hidden -translate-y-1/2 flex-col gap-2 lg:flex">
        {nav.map((n) => (
          <a key={n.id} href={`#${n.id}`} title={n.label} className="group flex h-3 items-center">
            <span className={`block h-px transition-all ${active === n.id ? "w-6 bg-ink" : "w-5 bg-line group-hover:bg-muted"}`} />
          </a>
        ))}
      </nav>
    </>
  );
}
