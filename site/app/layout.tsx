import type { Metadata } from "next";
import { Geist_Mono, Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "LensCraft",
  description:
    "An agent that simulates gravitational lenses, trains dark-matter substructure detectors on them, and uses what the models can't classify to decide what to simulate next.",
  openGraph: {
    title: "LensCraft",
    description: "Closing the simulate-train-resimulate loop. A case study built on Pydantic AI, Lenstronomy and Modal.",
    images: ["/figures/hero_classes.png"],
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} ${geistMono.variable} h-full`}>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
