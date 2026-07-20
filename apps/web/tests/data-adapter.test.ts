import { describe, expect, it } from "vitest";
import { getProjectSnapshot, resolveDataMode } from "@/src/data/adapter";

describe("dashboard data adapter", () => {
  it("uses authoritative repository mode by default", () => {
    const snapshot = getProjectSnapshot(resolveDataMode(undefined));
    expect(snapshot.source).toBe("repository");
    expect(snapshot.dataset.classCount).toBe(7);
    expect(snapshot.experiments).toHaveLength(5);
    expect(snapshot.dataset.splits).toEqual({ train: 427, val: 140, test: 68 });
  });

  it("never silently falls back from API mode", () => {
    expect(() => getProjectSnapshot("api")).toThrow(/API/);
  });

  it("keeps scientific results locked in repository mode", () => {
    const snapshot = getProjectSnapshot("repository");
    expect(snapshot.scientificResults.available).toBe(false);
    expect(snapshot.scientificResults.finalTestEvaluated).toBe(false);
    expect(snapshot.training.testSetAccessCount).toBe(0);
    expect(snapshot.audit.protectedContentIncluded).toBe(false);
  });
});
