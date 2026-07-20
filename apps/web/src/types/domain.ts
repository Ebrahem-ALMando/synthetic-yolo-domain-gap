export type DataMode = "repository" | "demo" | "api";
export type ScientificStatus =
  | "completed"
  | "in_progress"
  | "awaiting_results"
  | "protected"
  | "frozen"
  | "success"
  | "warning"
  | "failed"
  | "unavailable";

export interface ScientificIdentity {
  realSplit: string;
  syntheticPool: string;
  objectBank: string;
  generatorConfiguration: string;
  experimentDesign: string;
  training: string | null;
  evaluationContract: string;
  testCampaign: string;
}

export interface MetricSet {
  precision: number;
  recall: number;
  map50: number;
  map5095: number;
}

export interface ClassStatistic {
  id: number;
  key: string;
  nameAr: string;
  realTrainImages: number;
  syntheticObjects: number;
}

export interface ExperimentRegime {
  id: string;
  nameAr: string;
  realCount: number;
  syntheticCount: number;
  total: number;
  validationCount: number;
  realFraction: number;
  manifestHash: string;
  status: ScientificStatus;
  checkpointAvailable: boolean;
  checkpointHash: string;
  validationMetricsAvailable: boolean;
  finalTestMetricsAvailable: boolean;
  validationMetrics: MetricSet;
  finalMetrics: MetricSet;
  rank: number;
  recommended: boolean;
  bestEpoch: number;
  durationSeconds: number;
  latencyMs: number;
}

export interface ProjectSnapshot {
  schemaVersion: number;
  source: "repository" | "demo" | "api";
  gitRevision: string;
  project: {
    name: string;
    nameAr: string;
    descriptionAr: string;
    sloganAr: string;
    phase: string;
    dashboardStatus: string;
    seed: number;
  };
  identities: ScientificIdentity;
  dataset: {
    name: string;
    version: number;
    status: string;
    classCount: number;
    classes: ClassStatistic[];
    splits: { train: number; val: number; test: number };
    objects: { train: number; val: number; test: number };
    sourceGroups: { train: number; val: number; test: number };
    duplicateGroupCount: number;
    leakageStatus: string;
    protectedTest: boolean;
  };
  synthetic: {
    poolSize: number;
    status: string;
    mode: string;
    generatorVersion: string;
    pastedObjects: number;
    acceptedObjectBankItems: number;
    objectBankRecords: number;
    objectSizes: Record<string, number>;
    failedAttempts: number;
    rejectedPlacements: number;
  };
  experiments: ExperimentRegime[];
  training: {
    state: string;
    model: string;
    epochs: number;
    imageSize: number;
    preferredBatch: number;
    fallbackBatch: number;
    completedFinalRegimes: number;
    smokeRegimesCompleted: number;
    gpu: string | null;
    profile: string | null;
    testSetAccessCount: number;
    trainingIdentity: string;
  };
  environment: {
    python: string;
    torch: string;
    ultralytics: string;
    platform: string;
    cudaAvailableLocally: boolean;
    classification: string;
  };
  scientificResults: {
    available: boolean;
    finalTestEvaluated: boolean;
    messageAr: string;
    recommendedModel: string;
    primaryMetric: string;
    primaryMetricValue: number;
    ranking: Array<MetricSet & { rank: number; regime: string; recommended: boolean; latencyMs: number }>;
    perClass: Array<{
      regime: string;
      classId: number;
      className: string;
      precision: number;
      recall: number;
      ap50: number;
      ap5095: number;
    }>;
    objectSize: Array<{ regime: string; size: string; instances: number; map50: number; map5095: number }>;
    latency: Array<{
      regime: string;
      preprocessMs: number;
      inferenceMs: number;
      postprocessMs: number;
      totalMs: number;
      throughput: number;
    }>;
    domainGap: Array<{
      regime: string;
      realPercentage: number;
      map5095: number;
      absoluteChange: number;
      relativeChangePercent: number;
    }>;
    campaign: {
      id: string;
      attempt: string;
      status: string;
      successfulCampaigns: number;
      technicalFailures: number;
      contractHash: string;
    };
    resultHashes: Record<string, Record<string, string>>;
    errorSummary: {
      selectedCases: number;
      galleryAvailableLocally: boolean;
      galleryReasonAr: string;
      eventCounts: Record<string, Record<string, number>>;
    };
  };
  api: {
    implemented: boolean;
    baseUrl: string;
    modelsAvailableLocally: number;
    recommendedModel: string;
  };
  reports: Array<{ title: string; path: string }>;
  audit: {
    splitFrozen: boolean;
    syntheticFrozen: boolean;
    experimentValidationPassed: boolean;
    testSetUsedForExperiments: boolean;
    protectedContentIncluded: boolean;
    trainingTestAccessCount: number;
    authorizedEvaluationCampaigns: number;
    resultHashesVerified: number;
    predictionHashesVerified: number;
    intakeHashReportAvailable: boolean;
  };
  demoMetrics?: Array<{
    regime: string;
    precision: number;
    recall: number;
    map50: number;
    map5095: number;
  }>;
}

export interface AuditEvent {
  title: string;
  description: string;
  status: ScientificStatus;
}

export interface InferenceRequest {
  modelId: string;
  confidence: number;
  iou: number;
  imageSize: number;
  file: File;
}

export interface InferenceResponse {
  model_id: string;
  filename: string;
  original_width: number;
  original_height: number;
  detections: Array<{
    class_id: number;
    class_name: string;
    class_name_ar: string;
    confidence: number;
    bbox_xyxy_pixels: [number, number, number, number];
    bbox_xyxy_normalized: [number, number, number, number];
  }>;
  detection_count: number;
  preprocessing_duration_ms: number;
  inference_duration_ms: number;
  postprocessing_duration_ms: number;
  total_duration_ms: number;
  device: string;
  annotated_image_mime: string | null;
  annotated_image_base64: string | null;
}
