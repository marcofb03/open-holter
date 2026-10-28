package org.openholter.recorder

import android.content.Context
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import androidx.core.content.FileProvider
import org.openholter.core.EcgWindow
import org.openholter.core.HrBin
import org.openholter.core.Marker
import org.openholter.core.RecordingFiles
import org.openholter.core.Training
import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.ceil
import kotlin.math.floor

/**
 * PDF reports for a doctor, A4 landscape:
 * - ECG: standard ECG paper scale (25 mm/s, 10 mm/mV, 1 mm / 5 mm grid), 10 s
 *   per row, 4 rows per page, as many pages as needed; symptom markers as red
 *   lines. The scale is stated in the header (the owner prefers rows without a
 *   drawn calibration pulse). The signal is drawn as delivered by the H10.
 * - Heart-rate summary of a training: one page, time on x, bpm on y.
 *
 * Text uses the bundled static Roboto 2.138 (assets/fonts, Apache 2.0): with
 * the phone's variable system font, Android's PDF writer embeds Type 3 fonts,
 * from which PDF viewers copy text with missing or extra spaces.
 */
object ReportPdf {
    private const val PT_PER_MM = 72f / 25.4f
    private const val PAGE_W = 842 // A4 landscape in points
    private const val PAGE_H = 595
    private const val MM_PER_S = 25f
    private const val MM_PER_MV = 10f
    private const val SECONDS_PER_ROW = 10
    private const val ROW_MM = 35f
    private const val ROWS_PER_PAGE = 4
    private const val ROW_GAP_PT = 16f
    private const val TOP_PT = 112f
    private const val MARGIN = 40f

    /** Longest ECG export (about 270 pages); longer trainings get the summary only. */
    const val MAX_ECG_NS = 3 * 3_600_000_000_000L

    private val DATE_TIME: DateTimeFormatter = DateTimeFormatter.ofPattern("EEE d MMM yyyy, HH:mm:ss")
    private val CLOCK: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss")
    private val FILE_STAMP: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")
    private const val SIGNAL =
        "Polar H10 chest strap, single lead, 130 Hz. Signal as delivered by the H10, no extra filtering by the app " +
            "(the H10's internal filtering is not documented by Polar). Scale 25 mm/s, 10 mm/mV."
    private const val FOOTER = "Recorded with open-holter (open source). Not a medical device, not for diagnosis on its own."

    fun ecgPages(fromNs: Long, toNs: Long): Int =
        maxOf(1, ceil(ceil((toNs - fromNs) / (SECONDS_PER_ROW * 1e9)) / ROWS_PER_PAGE).toInt())

    fun marker(context: Context, marker: Marker, w: EcgWindow, fromNs: Long, toNs: Long, hrBpm: Int?): File {
        val zone = ZoneId.systemDefault()
        val beforeS = (marker.timestampNs - fromNs) / 1_000_000_000L
        val afterS = (toNs - marker.timestampNs) / 1_000_000_000L
        val info = buildList {
            add("ECG from $beforeS s before to $afterS s after the marker. $SIGNAL" + (hrBpm?.let { " H10 heart rate around the marker: $it bpm." } ?: ""))
            if (w.size == 0) add("No ECG was recorded around this marker (strap not connected or recording stopped).")
        }
        return ecg(
            context, "Symptom marker – ${DATE_TIME.format(local(marker.timestampNs, zone))} (${zone.id})",
            marker.comment, info, w, fromNs, toNs, listOf(marker.timestampNs),
            "open-holter_marker_${FILE_STAMP.format(local(marker.timestampNs, zone))}",
        )
    }

    fun trainingEcg(context: Context, t: Training, endNs: Long, w: EcgWindow, markers: List<Long>): File {
        val zone = ZoneId.systemDefault()
        val info = buildList {
            add("Whole training, ${duration(endNs - t.startNs)}. $SIGNAL" + if (markers.isNotEmpty()) " Red lines: symptom markers." else "")
            if (w.size == 0) add("No ECG was recorded during this training (strap not connected or recording stopped).")
        }
        return ecg(
            context, trainingTitle(t, endNs, zone), t.comment, info, w, t.startNs, endNs, markers,
            "open-holter_training_ecg_${FILE_STAMP.format(local(t.startNs, zone))}",
        )
    }

    /** One page: the H10 heart rate over the training ([bins] of [binNs]) and its numbers. */
    fun trainingSummary(context: Context, t: Training, endNs: Long, bins: List<HrBin>, binNs: Long, markers: List<Long>): File {
        val zone = ZoneId.systemDefault()
        val p = paints(context)
        val doc = PdfDocument()
        try {
            val page = doc.startPage(PdfDocument.PageInfo.Builder(PAGE_W, PAGE_H, 1).create())
            val c = page.canvas
            val info = buildList {
                add(
                    "Heart rate reported by the Polar H10 chest strap, median per ${binNs / 1_000_000_000L} s, " +
                        "only times with skin contact (the shaded band is the lowest and highest reading in each interval)." +
                        if (markers.isNotEmpty()) " Red lines: symptom markers." else "",
                )
                add(stats(bins, binNs, t.startNs, endNs))
            }
            header(c, p, trainingTitle(t, endNs, zone), t.comment, info)
            drawHrChart(c, p, bins, binNs, t.startNs, endNs, markers, zone, left = 70f, top = 125f, right = PAGE_W - MARGIN, bottom = PAGE_H - 60f)
            c.drawText("$FOOTER   Page 1 of 1", MARGIN, PAGE_H - 18f, p.small)
            doc.finishPage(page)
            return save(context, doc, "open-holter_training_summary_${FILE_STAMP.format(local(t.startNs, zone))}")
        } finally {
            doc.close()
        }
    }

    fun shareIntent(context: Context, file: File): Intent {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/pdf"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, file.nameWithoutExtension)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(send, "Share or save the report")
    }

    /** Lowest / median / highest of the bin medians, highest single reading, and coverage. */
    fun stats(bins: List<HrBin>, binNs: Long, fromNs: Long, toNs: Long): String {
        if (bins.isEmpty()) return "No heart rate with skin contact in this period."
        val med = bins.map { it.median }.sorted()
        val coveredMin = bins.size * binNs / 60e9
        return "Heart rate: lowest ${med.first()}, median ${med[med.size / 2]}, highest ${med.last()} bpm " +
            "(of the ${binNs / 1_000_000_000L}-s medians); highest single reading ${bins.maxOf { it.max }} bpm. " +
            "Data for %.0f of %.0f min.".format(coveredMin, (toNs - fromNs) / 60e9)
    }

    fun duration(ns: Long): String {
        val s = ns / 1_000_000_000L
        return if (s >= 3600) "%d h %02d min".format(s / 3600, s / 60 % 60) else "%d min %02d s".format(s / 60, s % 60)
    }

    // ---- ECG pages -------------------------------------------------------------------------

    private fun ecg(
        context: Context, title: String, comment: String, info: List<String>, w: EcgWindow,
        fromNs: Long, toNs: Long, markers: List<Long>, fileStem: String,
    ): File {
        val zone = ZoneId.systemDefault()
        val p = paints(context)
        val rowNs = SECONDS_PER_ROW * 1_000_000_000L
        val rows = ceil((toNs - fromNs).toDouble() / rowNs).toInt()
        val pages = ecgPages(fromNs, toNs)
        val doc = PdfDocument()
        try {
            for (pg in 0 until pages) {
                val page = doc.startPage(PdfDocument.PageInfo.Builder(PAGE_W, PAGE_H, pg + 1).create())
                val c = page.canvas
                header(c, p, title, comment, info)
                for (k in 0 until ROWS_PER_PAGE) {
                    val r = pg * ROWS_PER_PAGE + k
                    if (r >= rows) break
                    val top = TOP_PT + k * (ROW_MM * PT_PER_MM + ROW_GAP_PT)
                    drawRow(c, w, fromNs + r * rowNs, top, markers, zone, p.small)
                }
                c.drawText(
                    "$FOOTER Print at 100 % (no \"fit to page\") to keep 25 mm/s and 10 mm/mV.   Page ${pg + 1} of $pages",
                    MARGIN, PAGE_H - 18f, p.small,
                )
                doc.finishPage(page)
            }
            return save(context, doc, fileStem)
        } finally {
            doc.close()
        }
    }

    private fun drawRow(c: Canvas, w: EcgWindow, startNs: Long, top: Float, markers: List<Long>, zone: ZoneId, label: Paint) {
        val left = (PAGE_W - SECONDS_PER_ROW * MM_PER_S * PT_PER_MM) / 2
        val width = SECONDS_PER_ROW * MM_PER_S * PT_PER_MM
        val height = ROW_MM * PT_PER_MM
        val endNs = startNs + SECONDS_PER_ROW * 1_000_000_000L
        c.drawText(CLOCK.format(local(startNs, zone)), left, top - 3f, label)

        val minor = Paint().apply { color = Color.rgb(246, 201, 201); strokeWidth = 0.25f }
        val major = Paint().apply { color = Color.rgb(229, 154, 154); strokeWidth = 0.6f }
        val mmX = (SECONDS_PER_ROW * MM_PER_S).toInt()
        for (i in 0..mmX) {
            val x = left + i * PT_PER_MM
            c.drawLine(x, top, x, top + height, if (i % 5 == 0) major else minor)
        }
        for (i in 0..ROW_MM.toInt()) {
            val y = top + i * PT_PER_MM
            c.drawLine(left, y, left + width, y, if (i % 5 == 0) major else minor)
        }

        val baseY = top + height * 0.6f // room for 2.1 mV up, 1.4 mV down
        val i0 = w.t.binarySearch(startNs).let { if (it < 0) -it - 1 else it }
        val i1 = w.t.binarySearch(endNs).let { if (it < 0) -it - 1 else it }
        if (i1 > i0) {
            val median = w.mv.copyOfRange(i0, i1).sortedArray()[(i1 - i0) / 2]
            val path = Path()
            var prev = Long.MIN_VALUE
            for (i in i0 until i1) {
                val x = left + (w.t[i] - startNs) / 1e9f * MM_PER_S * PT_PER_MM
                val y = baseY - (w.mv[i] - median) * MM_PER_MV * PT_PER_MM
                if (prev == Long.MIN_VALUE || w.t[i] - prev > 50_000_000L) path.moveTo(x, y) else path.lineTo(x, y)
                prev = w.t[i]
            }
            val ink = Paint().apply {
                color = Color.BLACK; style = Paint.Style.STROKE; strokeWidth = 0.7f
                isAntiAlias = true; strokeJoin = Paint.Join.ROUND
            }
            c.save()
            c.clipRect(left, top, left + width, top + height)
            c.drawPath(path, ink)
            c.restore()
        } else {
            c.drawText("no ECG in this part", left + 6f, top + height / 2, label)
        }
        for (m in markers) {
            if (m !in startNs until endNs) continue
            val x = left + (m - startNs) / 1e9f * MM_PER_S * PT_PER_MM
            val red = Paint().apply {
                color = Color.rgb(211, 47, 47); strokeWidth = 1.5f; textSize = 8f; typeface = label.typeface; isAntiAlias = true
            }
            c.drawLine(x, top, x, top + height, red)
            c.drawText("MARKER ${CLOCK.format(local(m, zone))}", x + 3f, top + 9f, red)
        }
    }

    // ---- heart-rate chart ------------------------------------------------------------------

    private fun drawHrChart(
        c: Canvas, p: Paints, bins: List<HrBin>, binNs: Long, fromNs: Long, toNs: Long, markers: List<Long>, zone: ZoneId,
        left: Float, top: Float, right: Float, bottom: Float,
    ) {
        val gridPaint = Paint().apply { color = Color.rgb(210, 210, 210); strokeWidth = 0.5f }
        val axis = Paint().apply { color = Color.DKGRAY; strokeWidth = 0.8f }
        val w = right - left
        val h = bottom - top
        val medians = bins.map { it.median }
        var lo = if (medians.isEmpty()) 40 else floor((medians.min() - 10) / 10.0).toInt() * 10
        var hi = if (medians.isEmpty()) 180 else ceil((medians.max() + 10) / 10.0).toInt() * 10
        if (hi - lo < 40) {
            val mid = (hi + lo) / 2
            lo = mid - 20
            hi = mid + 20
        }
        val step = if (hi - lo > 100) 20 else 10
        val xOf = { t: Long -> left + (t - fromNs).toFloat() / (toNs - fromNs) * w }
        val yOf = { bpm: Int -> bottom - (bpm - lo).toFloat() / (hi - lo) * h }

        var g = lo
        while (g <= hi) {
            val y = yOf(g)
            c.drawLine(left, y, right, y, gridPaint)
            c.drawText("$g", left - 22f, y + 3f, p.small)
            g += step
        }
        c.drawText("bpm", left - 22f, top - 8f, p.small)
        val spanMin = (toNs - fromNs) / 60e9
        val tickMin = listOf(1, 2, 5, 10, 15, 30, 60, 120, 180, 240).firstOrNull { spanMin / it <= 14 } ?: 360
        var tick = Instant.ofEpochSecond(fromNs / 1_000_000_000L).atZone(zone).withSecond(0).withNano(0)
        val off = (tick.hour * 60 + tick.minute) % tickMin
        if (off != 0) tick = tick.plusMinutes((tickMin - off).toLong())
        while (true) {
            val t = tick.toInstant().epochSecond * 1_000_000_000L
            if (t > toNs) break
            if (t >= fromNs) {
                val x = xOf(t)
                c.drawLine(x, top, x, bottom, gridPaint)
                c.drawText("%02d:%02d".format(tick.hour, tick.minute), x - 11f, bottom + 12f, p.small)
            }
            tick = tick.plusMinutes(tickMin.toLong())
        }
        c.drawLine(left, bottom, right, bottom, axis)
        c.drawLine(left, top, left, bottom, axis)

        val blue = Color.rgb(21, 101, 192)
        val binW = maxOf(0.6f, w * binNs.toFloat() / (toNs - fromNs))
        val band = Paint().apply { color = blue; alpha = 60; strokeWidth = binW }
        for (b in bins) {
            val x = xOf(b.startNs) + binW / 2
            c.drawLine(x, yOf(b.max.coerceAtMost(hi)), x, yOf(b.min.coerceAtLeast(lo)), band)
        }
        val line = Paint().apply {
            color = blue; style = Paint.Style.STROKE; strokeWidth = 1.4f; isAntiAlias = true; strokeJoin = Paint.Join.ROUND
        }
        val path = Path()
        var prev = Long.MIN_VALUE
        for (b in bins) {
            val x = xOf(b.startNs) + binW / 2
            val y = yOf(b.median.coerceIn(lo, hi))
            if (prev == Long.MIN_VALUE || b.startNs - prev > maxOf(2 * binNs, 30_000_000_000L)) path.moveTo(x, y) else path.lineTo(x, y)
            prev = b.startNs
        }
        c.drawPath(path, line)
        val red = Paint().apply { color = Color.rgb(211, 47, 47); strokeWidth = 1.2f; textSize = 8f; typeface = p.small.typeface; isAntiAlias = true }
        for (m in markers) {
            if (m !in fromNs..toNs) continue
            val x = xOf(m)
            c.drawLine(x, top, x, bottom, red)
            c.drawText("MARKER ${CLOCK.format(local(m, zone))}", x + 3f, top + 9f, red)
        }
        if (bins.isEmpty()) c.drawText("No heart rate with skin contact in this period.", left + 10f, top + h / 2, p.text)
    }

    // ---- common ----------------------------------------------------------------------------

    private class Paints(val title: Paint, val text: Paint, val small: Paint)

    private fun paints(context: Context): Paints {
        val regular = font(context, "fonts/Roboto-Regular.ttf", Typeface.NORMAL)
        val bold = font(context, "fonts/Roboto-Bold.ttf", Typeface.BOLD)
        return Paints(
            title = Paint().apply { textSize = 14f; typeface = bold; isAntiAlias = true },
            text = Paint().apply { textSize = 9.5f; typeface = regular; isAntiAlias = true },
            small = Paint().apply { textSize = 8f; typeface = regular; color = Color.DKGRAY; isAntiAlias = true },
        )
    }

    /** Title, the comment (at most 2 lines) and the description (always shown), at most 5 lines in all. */
    private fun header(c: Canvas, p: Paints, title: String, comment: String, info: List<String>) {
        val maxW = PAGE_W - 2 * MARGIN
        val commentLines = wrap("Comment: " + comment.ifBlank { "(none)" }, p.text, maxW).let {
            if (it.size > 2) listOf(it[0], it[1].dropLast(1) + "…") else it
        }
        val lines = commentLines + info.flatMap { wrap(it, p.text, maxW) }
        c.drawText(title, MARGIN, 40f, p.title)
        lines.take(5).forEachIndexed { i, line -> c.drawText(line, MARGIN, 56f + i * 11.5f, p.text) }
    }

    private fun trainingTitle(t: Training, endNs: Long, zone: ZoneId): String =
        "Training – ${DATE_TIME.format(local(t.startNs, zone))} to ${CLOCK.format(local(endNs, zone))} " +
            "(${duration(endNs - t.startNs)}, ${zone.id})" + if (t.running) ", still running" else ""

    private fun save(context: Context, doc: PdfDocument, stem: String): File {
        val dir = File(context.cacheDir, "reports").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() } // only the latest export is kept
        val file = File(dir, "$stem.pdf")
        file.outputStream().use { doc.writeTo(it) }
        return file
    }

    private fun local(ns: Long, zone: ZoneId) = RecordingFiles.instantOf(ns).atZone(zone)

    private fun font(context: Context, asset: String, fallbackStyle: Int): Typeface =
        runCatching { Typeface.createFromAsset(context.assets, asset) }
            .getOrElse { Typeface.defaultFromStyle(fallbackStyle) }

    /** Splits [s] into lines no wider than [maxWidth] (by words). */
    private fun wrap(s: String, paint: Paint, maxWidth: Float): List<String> {
        val out = mutableListOf<String>()
        var line = StringBuilder()
        for (word in s.split(' ')) {
            val candidate = if (line.isEmpty()) word else "$line $word"
            if (paint.measureText(candidate) > maxWidth && line.isNotEmpty()) {
                out += line.toString()
                line = StringBuilder(word)
            } else {
                line = StringBuilder(candidate)
            }
        }
        if (line.isNotEmpty()) out += line.toString()
        return out
    }
}
