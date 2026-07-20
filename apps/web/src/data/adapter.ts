import { demoSnapshot } from "@/src/data/demo";
import { repositorySnapshot } from "@/src/data/repository";
import type { DataMode, ProjectSnapshot } from "@/src/types/domain";

export function resolveDataMode(value = process.env.NEXT_PUBLIC_DATA_MODE): DataMode {
  if (!value || value === "repository") return "repository";
  if (value === "demo" || value === "api") return value;
  throw new Error(`NEXT_PUBLIC_DATA_MODE غير مدعوم: ${value}`);
}

export function getProjectSnapshot(mode = resolveDataMode()): ProjectSnapshot {
  if (mode === "repository") return repositorySnapshot;
  if (mode === "demo") return demoSnapshot;
  throw new Error("وضع API يتطلب نقطة نهاية صريحة؛ لا يوجد رجوع تلقائي إلى بيانات تجريبية.");
}
