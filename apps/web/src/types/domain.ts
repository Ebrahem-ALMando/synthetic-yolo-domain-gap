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
  validationMetricsAvailable: boolean;
  finalTestMetricsAvailable: boolean;
}

export interface ProjectSnapshot {
  schemaVersion: number;
  source: "repository" | "demo";
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
  };
  audit: {
    splitFrozen: boolean;
    syntheticFrozen: boolean;
    experimentValidationPassed: boolean;
    testSetUsedForExperiments: boolean;
    protectedContentIncluded: boolean;
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
  available: boolean;
  detections: Array<{ className: string; confidence: number; box: [number, number, number, number] }>;
  processingMs: number | null;
  source: "model" | "demo";
}
