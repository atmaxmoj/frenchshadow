"use client";

import { installGlobalClientLogging } from "@/lib/clientLogger";
import { useEffect } from "react";

export function ClientLogInit() {
  useEffect(() => {
    installGlobalClientLogging();
  }, []);
  return null;
}
