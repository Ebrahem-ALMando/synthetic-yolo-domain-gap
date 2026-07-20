import { DashboardPage } from "@/src/components/dashboard-pages";
import { loadProjectSnapshot, resolveDataMode } from "@/src/data/adapter";

export async function RepositoryRoute({ path }: { path: string }) {
  const snapshot = await loadProjectSnapshot(resolveDataMode());
  return <DashboardPage snapshot={snapshot} path={path} />;
}
