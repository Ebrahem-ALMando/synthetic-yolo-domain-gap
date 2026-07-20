import { afterEach, describe, expect, it, vi } from "vitest";
import { getProjectSnapshot, loadProjectSnapshot, resolveDataMode } from "@/src/data/adapter";

afterEach(() => vi.unstubAllGlobals());

describe("dashboard data adapter", () => {
  it("uses authoritative repository mode by default", () => {
    const snapshot = getProjectSnapshot(resolveDataMode(undefined));
    expect(snapshot.source).toBe("repository");
    expect(snapshot.dataset.classCount).toBe(7);
    expect(snapshot.experiments).toHaveLength(5);
    expect(snapshot.dataset.splits).toEqual({ train: 427, val: 140, test: 68 });
  });

  it("never silently falls back from API mode", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    await expect(loadProjectSnapshot("api")).rejects.toThrow(/503/);
  });

  it("loads sealed scientific results in repository mode", () => {
    const snapshot = getProjectSnapshot("repository");
    expect(snapshot.scientificResults.available).toBe(true);
    expect(snapshot.scientificResults.finalTestEvaluated).toBe(true);
    expect(snapshot.scientificResults.recommendedModel).toBe("real_only");
    expect(snapshot.scientificResults.ranking).toHaveLength(5);
    expect(snapshot.training.testSetAccessCount).toBe(0);
    expect(snapshot.audit.authorizedEvaluationCampaigns).toBe(1);
    expect(snapshot.audit.protectedContentIncluded).toBe(false);
  });
});
