import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://localhost:8767/:path*" }];
  },
};

export default nextConfig;
