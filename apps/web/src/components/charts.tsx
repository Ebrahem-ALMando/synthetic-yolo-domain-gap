"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  Radar,
  RadarChart,
  PolarAngleAxis,
  PolarGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ClassStatistic, ExperimentRegime, ProjectSnapshot } from "@/src/types/domain";
import { formatNumber } from "@/src/lib/utils";

const colors = ["#2563EB", "#06B6D4", "#38BDF8", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444"];

const tooltipStyle = { borderRadius: 12, borderColor: "#E2E8F0", direction: "rtl" as const };

export function ClassDistributionChart({ classes }: { classes: ClassStatistic[] }) {
  return <div className="h-72" role="img" aria-label="مخطط توزيع صور الفئات في التدريب الحقيقي"><ResponsiveContainer width="100%" height="100%"><BarChart data={classes} margin={{ top: 8, right: 0, left: 0, bottom: 8 }}><CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.25} /><XAxis dataKey="nameAr" tick={{ fontSize: 11 }} interval={0} /><YAxis orientation="right" tick={{ fontSize: 11 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => [formatNumber.format(Number(value)), "صور التدريب"]} /><Bar dataKey="realTrainImages" radius={[6, 6, 0, 0]}>{classes.map((item, index) => <Cell key={item.key} fill={colors[index]} />)}</Bar></BarChart></ResponsiveContainer></div>;
}

export function RegimeCompositionChart({ regimes }: { regimes: ExperimentRegime[] }) {
  return <div className="h-72" role="img" aria-label="نسبة الصور الحقيقية إلى الاصطناعية في التجارب الخمس"><ResponsiveContainer width="100%" height="100%"><BarChart data={regimes} layout="vertical" margin={{ top: 8, right: 4, left: 8, bottom: 8 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.25} /><XAxis type="number" domain={[0, 427]} tick={{ fontSize: 11 }} /><YAxis type="category" dataKey="nameAr" orientation="right" width={82} tick={{ fontSize: 11 }} /><Tooltip contentStyle={tooltipStyle} /><Legend /><Bar name="حقيقي" dataKey="realCount" stackId="a" fill="#2563EB" /><Bar name="اصطناعي" dataKey="syntheticCount" stackId="a" fill="#06B6D4" radius={[6, 0, 0, 6]} /></BarChart></ResponsiveContainer></div>;
}

export function ObjectSizeChart({ sizes }: { sizes: Record<string, number> }) {
  const labels: Record<string, string> = { small: "صغير", medium: "متوسط", large: "كبير" };
  const data = Object.entries(sizes).map(([key, value]) => ({ name: labels[key] ?? key, value }));
  return <div className="h-72" role="img" aria-label="توزيع أحجام أجسام بنك البيانات الاصطناعي"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={data} dataKey="value" nameKey="name" innerRadius={62} outerRadius={96} paddingAngle={3}>{data.map((item, index) => <Cell key={item.name} fill={colors[index]} />)}</Pie><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatNumber.format(Number(value))} /><Legend /></PieChart></ResponsiveContainer></div>;
}

export function DemoMetricCharts({ snapshot }: { snapshot: ProjectSnapshot }) {
  const data = snapshot.demoMetrics ?? [];
  return <div className="grid gap-4 xl:grid-cols-2"><div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={data}><CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.25} /><XAxis dataKey="regime" tick={{ fontSize: 11 }} /><YAxis orientation="right" domain={[0, 1]} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => Number(value).toFixed(3)} /><Legend /><Bar dataKey="precision" name="Precision" fill="#2563EB" radius={[5, 5, 0, 0]} /><Bar dataKey="recall" name="Recall" fill="#06B6D4" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div><div className="h-72"><ResponsiveContainer width="100%" height="100%"><RadarChart data={data}><PolarGrid /><PolarAngleAxis dataKey="regime" tick={{ fontSize: 11 }} /><Radar name="mAP@50" dataKey="map50" stroke="#2563EB" fill="#2563EB" fillOpacity={0.25} /><Radar name="mAP@50–95" dataKey="map5095" stroke="#06B6D4" fill="#06B6D4" fillOpacity={0.18} /><Legend /><Tooltip contentStyle={tooltipStyle} /></RadarChart></ResponsiveContainer></div></div>;
}

export function FinalMetricCharts({ snapshot }: { snapshot: ProjectSnapshot }) {
  const ranking = snapshot.scientificResults.ranking.map((row) => ({
    ...row,
    name: snapshot.experiments.find((regime) => regime.id === row.regime)?.nameAr ?? row.regime,
  }));
  return <div className="grid gap-4 xl:grid-cols-2"><div className="h-80"><ResponsiveContainer width="100%" height="100%"><BarChart data={ranking}><CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.25} /><XAxis dataKey="name" tick={{ fontSize: 11 }} /><YAxis orientation="right" domain={[0, .7]} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => Number(value).toFixed(4)} /><Legend /><Bar dataKey="precision" name="Precision" fill="#2563EB" radius={[5, 5, 0, 0]} /><Bar dataKey="recall" name="Recall" fill="#06B6D4" radius={[5, 5, 0, 0]} /><Bar dataKey="map5095" name="mAP@50–95" fill="#10B981" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div><div className="h-80"><ResponsiveContainer width="100%" height="100%"><LineChart data={snapshot.scientificResults.domainGap}><CartesianGrid strokeDasharray="3 3" opacity={0.25} /><XAxis dataKey="realPercentage" tickFormatter={(value) => `${Number(value).toFixed(0)}%`} /><YAxis orientation="right" domain={[.15, .23]} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => Number(value).toFixed(5)} labelFormatter={(value) => `${Number(value).toFixed(1)}% بيانات حقيقية`} /><Line type="monotone" dataKey="map5095" name="mAP@50–95" stroke="#2563EB" strokeWidth={3} dot={{ r: 5 }} /></LineChart></ResponsiveContainer></div></div>;
}
