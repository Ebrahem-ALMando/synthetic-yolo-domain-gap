import { repositorySnapshot } from "@/src/data/repository";
import type { ProjectSnapshot } from "@/src/types/domain";

// Presentation-only values. They are intentionally isolated and always accompanied by DemoDataBanner.
export const demoSnapshot: ProjectSnapshot = {
  ...repositorySnapshot,
  source: "demo",
  demoMetrics: repositorySnapshot.experiments.map((regime, index) => ({
    regime: regime.nameAr,
    precision: 0.63 + index * 0.04,
    recall: 0.58 + index * 0.045,
    map50: 0.61 + index * 0.035,
    map5095: 0.36 + index * 0.032,
  })),
};
