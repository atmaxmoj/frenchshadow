import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lucerne · Shadow Reader",
  description: "YouTube 影子跟读 — 逐句跟读，即时法语发音点评。",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
