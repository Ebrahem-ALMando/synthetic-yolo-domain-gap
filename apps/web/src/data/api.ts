import { repositorySnapshot } from "@/src/data/repository";
import type { MetricSet, ProjectSnapshot } from "@/src/types/domain";

type ApiMetricSet = {
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
};

type ApiModel = {
  model_id: string;
  available: boolean;
  checkpoint_sha256: string;
  recommended: boolean;
  final_metrics: ApiMetricSet;
};

type ApiEvaluationModel = {
  regime: string;
  metrics: ApiMetricSet;
  latency_ms_per_image: number;
};

type ApiPayloads = {
  health: { status: string };
  project: {
    project_status: string;
    phase: string;
    models_completed: number;
    recommended_model: string;
    primary_metric_value: number;
    campaign_id: string;
  };
  models: { models: ApiModel[]; recommended_model: string };
  evaluation: {
    campaign_id: string;
    attempt_id: string;
    status: string;
    recommended_model: string;
    ranking: Array<Record<string, string>>;
    models: ApiEvaluationModel[];
  };
  training: {
    status: string;
    training_identity: string;
    profile: { name: string; image_size: number; batch: number; gpu: string };
    test_access_count: number;
  };
  reproducibility: {
    repository_revision: string;
    contract_sha256: string;
    training_identity: string;
    campaign_id: string;
    successful_campaign_count: number;
    failed_technical_attempt_count: number;
  };
  reports: { reports: Array<{ title: string; repository_path: string }> };
};

function metricSet(metrics: ApiMetricSet): MetricSet {
  return {
    precision: Number(metrics.precision),
    recall: Number(metrics.recall),
    map50: Number(metrics.map50),
    map5095: Number(metrics.map50_95),
  };
}

async function request<T>(baseUrl: string, path: string): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`SynthDet API ${path}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export async function getApiProjectSnapshot(): Promise<ProjectSnapshot> {
  const baseUrl = (process.env.SYNTHDET_API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
  const [health, project, models, evaluation, training, reproducibility, reports] = await Promise.all([
    request<ApiPayloads["health"]>(baseUrl, "/health"),
    request<ApiPayloads["project"]>(baseUrl, "/api/v1/project"),
    request<ApiPayloads["models"]>(baseUrl, "/api/v1/models"),
    request<ApiPayloads["evaluation"]>(baseUrl, "/api/v1/evaluation"),
    request<ApiPayloads["training"]>(baseUrl, "/api/v1/training"),
    request<ApiPayloads["reproducibility"]>(baseUrl, "/api/v1/reproducibility"),
    request<ApiPayloads["reports"]>(baseUrl, "/api/v1/reports"),
  ]);

  const expected = repositorySnapshot.scientificResults;
  if (
    health.status !== "ok" ||
    project.models_completed !== 5 ||
    models.models.length !== 5 ||
    evaluation.models.length !== 5 ||
    project.campaign_id !== expected.campaign.id ||
    evaluation.campaign_id !== expected.campaign.id ||
    reproducibility.campaign_id !== expected.campaign.id ||
    evaluation.recommended_model !== expected.recommendedModel
  ) {
    throw new Error("SynthDet API identity does not match the sealed repository snapshot");
  }

  const modelById = new Map(models.models.map((model) => [model.model_id, model]));
  const evaluationById = new Map(evaluation.models.map((model) => [model.regime, model]));
  const ranking = evaluation.ranking.map((row) => ({
    rank: Number(row.rank),
    regime: row.regime,
    recommended: row.recommended.toLowerCase() === "true",
    precision: Number(row.precision),
    recall: Number(row.recall),
    map50: Number(row.map50),
    map5095: Number(row.map50_95),
    latencyMs: Number(row.latency_ms),
  }));
  const rankById = new Map(ranking.map((row) => [row.regime, row.rank]));

  return {
    ...repositorySnapshot,
    source: "api",
    gitRevision: reproducibility.repository_revision,
    project: {
      ...repositorySnapshot.project,
      phase: project.phase,
      dashboardStatus: project.project_status,
    },
    identities: {
      ...repositorySnapshot.identities,
      training: reproducibility.training_identity,
      evaluationContract: reproducibility.contract_sha256,
      testCampaign: reproducibility.campaign_id,
    },
    experiments: repositorySnapshot.experiments.map((regime) => {
      const model = modelById.get(regime.id);
      const result = evaluationById.get(regime.id);
      if (!model || !result) throw new Error(`SynthDet API is missing regime ${regime.id}`);
      return {
        ...regime,
        checkpointAvailable: model.available,
        checkpointHash: model.checkpoint_sha256,
        finalMetrics: metricSet(result.metrics),
        latencyMs: Number(result.latency_ms_per_image),
        rank: rankById.get(regime.id) ?? regime.rank,
        recommended: model.recommended,
      };
    }),
    training: {
      ...repositorySnapshot.training,
      state: training.status,
      imageSize: training.profile.image_size,
      preferredBatch: training.profile.batch,
      gpu: training.profile.gpu,
      profile: training.profile.name,
      testSetAccessCount: training.test_access_count,
      trainingIdentity: training.training_identity,
    },
    scientificResults: {
      ...repositorySnapshot.scientificResults,
      recommendedModel: project.recommended_model,
      primaryMetricValue: Number(project.primary_metric_value),
      ranking,
      campaign: {
        ...repositorySnapshot.scientificResults.campaign,
        id: evaluation.campaign_id,
        attempt: evaluation.attempt_id,
        status: evaluation.status,
        successfulCampaigns: reproducibility.successful_campaign_count,
        technicalFailures: reproducibility.failed_technical_attempt_count,
      },
    },
    api: {
      implemented: true,
      baseUrl,
      modelsAvailableLocally: models.models.filter((model) => model.available).length,
      recommendedModel: models.recommended_model,
    },
    reports: reports.reports.map((report) => ({ title: report.title, path: report.repository_path })),
  };
}
