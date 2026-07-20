"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import * as Slider from "@radix-ui/react-slider";
import { Download, ImageOff, Play, RotateCcw, ShieldCheck, UploadCloud } from "lucide-react";
import Image from "next/image";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button, Card, StatusBadge, TechnicalValue } from "@/src/components/ui";
import type { ExperimentRegime, InferenceResponse, ProjectSnapshot } from "@/src/types/domain";

const schema = z.object({
  modelId: z.string().min(1),
  confidence: z.number().min(0.001).max(1),
  iou: z.number().min(0.1).max(0.95),
});
type FormValues = z.infer<typeof schema>;

function RangeField({
  label,
  value,
  min = 0,
  max = 1,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="mb-2 flex justify-between text-sm font-bold">
        <span>{label}</span>
        <bdi dir="ltr" className="technical-ltr">{value.toFixed(2)}</bdi>
      </span>
      <Slider.Root
        value={[value]}
        min={min}
        max={max}
        step={0.01}
        onValueChange={([next]) => onChange(next)}
        className="relative flex h-5 touch-none items-center"
      >
        <Slider.Track className="relative h-2 grow rounded-full bg-muted">
          <Slider.Range className="absolute h-full rounded-full bg-primary" />
        </Slider.Track>
        <Slider.Thumb
          className="block h-5 w-5 rounded-full border-2 border-primary bg-card shadow"
          aria-label={label}
        />
      </Slider.Root>
    </label>
  );
}

export function InferenceWorkspace({ regimes }: { regimes: ExperimentRegime[] }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const recommended = regimes.find((regime) => regime.recommended)?.id ?? regimes[0].id;
  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  const { register, watch, setValue, reset, handleSubmit } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { modelId: recommended, confidence: 0.25, iou: 0.7 },
  });
  const values = watch();

  function selectFile(selected?: File) {
    if (!selected) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }

  function clear() {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setFile(null);
    setResult(null);
    setError(null);
    reset({ modelId: recommended, confidence: 0.25, iou: 0.7 });
  }

  async function submit(form: FormValues) {
    if (!file) {
      setError("اختر صورة خارجية أولًا.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    const body = new FormData();
    body.set("file", file);
    body.set("model_id", form.modelId);
    body.set("confidence", String(form.confidence));
    body.set("iou", String(form.iou));
    body.set("max_detections", "100");
    body.set("annotate", "true");
    try {
      const response = await fetch(`${apiBase}/api/v1/inference`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.message ?? `API ${response.status}`);
      setResult(payload as InferenceResponse);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "تعذر الاتصال بخدمة الاستدلال.");
    } finally {
      setLoading(false);
    }
  }

  const annotated = result?.annotated_image_base64
    ? `data:${result.annotated_image_mime};base64,${result.annotated_image_base64}`
    : null;

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1.5fr)_22rem]">
      <div className="min-w-0 space-y-4">
        <Card className="overflow-hidden p-4">
          <div className="grid gap-3 md:grid-cols-2">
            <div
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                selectFile(event.dataTransfer.files[0]);
              }}
              className="relative grid min-h-[360px] place-items-center overflow-hidden rounded-xl border border-dashed bg-muted/25"
            >
              {preview ? (
                <Image src={preview} alt="الصورة الأصلية" fill unoptimized className="object-contain" />
              ) : (
                <label className="cursor-pointer p-8 text-center">
                  <UploadCloud className="mx-auto h-14 w-14 text-primary" />
                  <b className="mt-4 block">اسحب صورة خارجية أو اخترها</b>
                  <span className="mt-2 block text-sm text-muted-foreground">
                    JPG أو PNG أو WebP — صور الاختبار المحمية مرفوضة بالبصمة
                  </span>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    className="sr-only"
                    onChange={(event) => selectFile(event.target.files?.[0])}
                  />
                </label>
              )}
              <span className="absolute right-3 top-3 rounded-lg bg-background/85 px-2 py-1 text-xs font-bold">
                الأصل
              </span>
            </div>
            <div className="relative grid min-h-[360px] place-items-center overflow-hidden rounded-xl border bg-slate-950/5">
              {annotated ? (
                <Image src={annotated} alt="نتيجة الكشف المعلّمة" fill unoptimized className="object-contain" />
              ) : (
                <div className="p-8 text-center text-muted-foreground">
                  <ImageOff className="mx-auto h-12 w-12" />
                  <p className="mt-3 text-sm">ستظهر نتيجة النموذج هنا</p>
                </div>
              )}
              <span className="absolute right-3 top-3 rounded-lg bg-background/85 px-2 py-1 text-xs font-bold">
                النتيجة
              </span>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <StatusBadge
              status={result ? "success" : error ? "failed" : "in_progress"}
              label={
                result
                  ? `${result.detection_count} كشف · ${result.total_duration_ms.toFixed(1)} ms · ${result.device}`
                  : error ?? "API صريح دون رجوع تجريبي"
              }
            />
            {annotated && (
                      <Button asChild variant="outline" className="h-8 px-3 text-xs">
                <a href={annotated} download={`${file?.name ?? "result"}-annotated.png`}>
                  <Download className="h-4 w-4" /> تنزيل النتيجة
                </a>
              </Button>
            )}
          </div>
        </Card>
        {result && (
          <Card className="overflow-x-auto p-4">
            <table className="w-full min-w-[620px] text-sm">
              <thead>
                <tr className="border-b text-right text-xs text-muted-foreground">
                  <th className="p-2">الفئة</th><th className="p-2">الثقة</th><th className="p-2">المربع (بكسل)</th>
                </tr>
              </thead>
              <tbody>
                {result.detections.map((detection, index) => (
                  <tr key={`${detection.class_id}-${index}`} className="border-b last:border-0">
                    <td className="p-2 font-bold">
                      {detection.class_name_ar} <bdi dir="ltr" className="text-xs text-muted-foreground">({detection.class_name})</bdi>
                    </td>
                    <td className="p-2"><bdi dir="ltr">{(detection.confidence * 100).toFixed(1)}%</bdi></td>
                    <td className="p-2"><bdi dir="ltr" className="technical-ltr">{detection.bbox_xyxy_pixels.map((value) => value.toFixed(1)).join(", ")}</bdi></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>
      <form onSubmit={handleSubmit(submit)} className="min-w-0 space-y-4">
        <Card className="space-y-5 p-5">
          <div>
            <h2 className="font-extrabold">إعدادات الاستدلال</h2>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              الحجم ثابت عند 640 بكسل. عتبات المختبر لا تغيّر حملة الاختبار المختومة.
            </p>
          </div>
          <label className="block text-sm font-bold">
            النموذج
            <select {...register("modelId")} className="mt-2 h-11 w-full rounded-xl border bg-background px-3">
              {regimes.map((regime) => (
                <option key={regime.id} value={regime.id}>
                  {regime.nameAr}{regime.recommended ? " — الموصى به" : ""}
                </option>
              ))}
            </select>
          </label>
          <RangeField label="عتبة الثقة" value={values.confidence} min={0.001} onChange={(value) => setValue("confidence", value)} />
          <RangeField label="عتبة IoU" value={values.iou} min={0.1} max={0.95} onChange={(value) => setValue("iou", value)} />
          <div className="grid grid-cols-2 gap-2">
            <Button disabled={loading || !file}><Play className="h-4 w-4" />{loading ? "جارٍ الكشف…" : "تشغيل"}</Button>
            <Button type="button" variant="outline" onClick={clear}><RotateCcw className="h-4 w-4" />إعادة ضبط</Button>
          </div>
        </Card>
        <Card className="p-5">
          <div className="flex items-center justify-between">
            <h3 className="font-bold">نتيجة JSON</h3>
            <StatusBadge status={result ? "success" : "in_progress"} label={result ? "خرج حقيقي" : "بانتظار التشغيل"} />
          </div>
          <pre dir="ltr" className="mt-3 max-h-80 w-full max-w-full overflow-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-300">
            {JSON.stringify(result ?? { status: error ? "error" : "idle", message: error, endpoint: `${apiBase}/api/v1/inference` }, null, 2)}
          </pre>
        </Card>
      </form>
    </div>
  );
}

const errorLabels: Record<string, string> = {
  true_positive: "كشوف مطابقة",
  false_positive: "إيجابيات كاذبة",
  false_negative: "سلبيات كاذبة",
  localization_error: "أخطاء تموضع",
  class_confusion: "التباس فئات",
};

export function FailureAnalysisGallery({ snapshot }: { snapshot: ProjectSnapshot }) {
  const [model, setModel] = useState(snapshot.scientificResults.recommendedModel);
  const counts = snapshot.scientificResults.errorSummary.eventCounts[model] ?? {};
  return (
    <div className="space-y-4">
      <Card className="flex flex-col gap-4 p-5 lg:flex-row lg:items-center">
        <div className="grid h-12 w-12 place-items-center rounded-xl bg-primary/10 text-primary"><ShieldCheck /></div>
        <div className="flex-1">
          <h2 className="font-extrabold">تحليل حتمي بعد ختم الحملة</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {snapshot.scientificResults.errorSummary.selectedCases} حالة ميتاداتا مختارة بقواعد ثابتة؛ لا تُنشر بكسلات الاختبار في الواجهة.
          </p>
        </div>
        <label className="text-sm font-bold">
          النموذج
          <select value={model} onChange={(event) => setModel(event.target.value)} className="mr-2 h-10 rounded-xl border bg-background px-3">
            {snapshot.experiments.map((regime) => <option key={regime.id} value={regime.id}>{regime.nameAr}</option>)}
          </select>
        </label>
      </Card>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {Object.entries(errorLabels).map(([key, label]) => (
          <Card key={key} className="p-5">
            <p className="text-xs font-bold text-muted-foreground">{label}</p>
            <bdi dir="ltr" className="mt-3 block text-3xl font-extrabold">{counts[key] ?? 0}</bdi>
            <p className="mt-2 text-xs text-muted-foreground">عند أرضية الثقة المجمّدة 0.001</p>
          </Card>
        ))}
      </div>
      <Card className="grid min-h-56 place-items-center p-8 text-center">
        <div>
          <ImageOff className="mx-auto h-12 w-12 text-muted-foreground" />
          <h3 className="mt-4 font-extrabold">معرض الصور محجوب عن أصول الويب</h3>
          <p className="mt-2 max-w-xl text-sm text-muted-foreground">{snapshot.scientificResults.errorSummary.galleryReasonAr}</p>
          <div className="mt-4"><TechnicalValue value="reports/analysis/error_cases.csv" copy={false} /></div>
        </div>
      </Card>
    </div>
  );
}
