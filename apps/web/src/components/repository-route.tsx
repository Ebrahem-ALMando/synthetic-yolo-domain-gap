import { DashboardPage } from "@/src/components/dashboard-pages";
import { getProjectSnapshot, resolveDataMode } from "@/src/data/adapter";

export function RepositoryRoute({ path }: { path: string }) {
  return <DashboardPage snapshot={getProjectSnapshot(resolveDataMode())} path={path} />;
}
