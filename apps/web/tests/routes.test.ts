import { describe, expect, it } from "vitest";
import { appRoutes, normalizePath, routeForPath } from "@/src/lib/routes";

describe("Sprint 6A route registry", () => {
  it("declares all twelve primary Arabic routes", () => {
    expect(appRoutes).toHaveLength(12);
    expect(appRoutes.map((route) => route.href)).toEqual([
      "/", "/experiments", "/datasets/real", "/datasets/synthetic", "/training",
      "/evaluation", "/analysis", "/inference", "/reports", "/reproducibility",
      "/system", "/about",
    ]);
    expect(appRoutes.every((route) => /[\u0600-\u06FF]/.test(route.title))).toBe(true);
  });

  it("resolves dynamic experiment pages to the experiment experience", () => {
    expect(routeForPath("/experiments/real_50")?.href).toBe("/experiments");
    expect(normalizePath(["datasets", "real"])).toBe("/datasets/real");
  });
});
