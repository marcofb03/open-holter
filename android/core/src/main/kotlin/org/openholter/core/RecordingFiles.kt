package org.openholter.core

import java.io.File
import java.time.Instant
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.temporal.ChronoUnit

/**
 * On-disk layout of a recording:
 *
 * ```
 * <root>/2026-10-21/ecg_2026-10-21_14.csv     ECG + HR/RR + markers (ECGLogger layout)
 * <root>/2026-10-21/acc_2026-10-21_14.csv     H10 accelerometer (signal-quality evidence)
 * <root>/2026-10-21/hr_2026-10-21_14.csv      H10 heart rate, RR and skin contact
 * <root>/2026-10-21/events_2026-10-21.log     connection / pause / gap log
 * <root>/2026-10-21/markers_2026-10-21.csv    symptom markers with optional comment
 * <root>/2026-10-21/trainings_2026-10-21.csv  training sessions (start, end, comment)
 * ```
 *
 * Files are split by the local wall-clock hour of the data they contain.
 */
object RecordingFiles {
    /** Same columns as ECGLogger, so `analysis/ecgpipe` can read the files directly. */
    const val ECG_HEADER = "timestamp_ns,ecg_mV,pulse_bpm,rr_ms,marker"
    const val ACC_HEADER = "timestamp_ns,x_mG,y_mG,z_mG"
    const val HR_HEADER = "timestamp_ns,hr_bpm,rr_ms,contact,contact_supported"
    const val EVENTS_HEADER = "time_local\ttimestamp_ns\tevent\tdetail"
    const val MARKERS_HEADER = "timestamp_ns,time_local,comment"
    const val TRAININGS_HEADER = "start_ns,end_ns,start_local,end_local,comment"

    fun hourOf(timestampNs: Long, zone: ZoneId): LocalDateTime =
        LocalDateTime.ofInstant(instantOf(timestampNs), zone).truncatedTo(ChronoUnit.HOURS)

    fun dayOf(timestampNs: Long, zone: ZoneId): LocalDate =
        LocalDateTime.ofInstant(instantOf(timestampNs), zone).toLocalDate()

    fun dayDir(root: File, day: LocalDate): File = File(root, day.toString())

    fun ecgFile(root: File, hour: LocalDateTime): File =
        File(dayDir(root, hour.toLocalDate()), "ecg_${hour.toLocalDate()}_${"%02d".format(hour.hour)}.csv")

    fun accFile(root: File, hour: LocalDateTime): File =
        File(dayDir(root, hour.toLocalDate()), "acc_${hour.toLocalDate()}_${"%02d".format(hour.hour)}.csv")

    fun hrFile(root: File, hour: LocalDateTime): File =
        File(dayDir(root, hour.toLocalDate()), "hr_${hour.toLocalDate()}_${"%02d".format(hour.hour)}.csv")

    fun eventsFile(root: File, day: LocalDate): File =
        File(dayDir(root, day), "events_$day.log")

    fun markersFile(root: File, day: LocalDate): File =
        File(dayDir(root, day), "markers_$day.csv")

    /** Training sessions, filed under the day they started. */
    fun trainingsFile(root: File, day: LocalDate): File =
        File(dayDir(root, day), "trainings_$day.csv")

    /** Day folders under [root], newest first. */
    fun listDays(root: File): List<LocalDate> =
        root.listFiles().orEmpty()
            .filter { it.isDirectory }
            .mapNotNull { runCatching { LocalDate.parse(it.name) }.getOrNull() }
            .sortedDescending()

    fun instantOf(timestampNs: Long): Instant =
        Instant.ofEpochSecond(Math.floorDiv(timestampNs, 1_000_000_000L), Math.floorMod(timestampNs, 1_000_000_000L))

    fun nowNs(): Long {
        val now = Instant.now()
        return now.epochSecond * 1_000_000_000L + now.nano
    }
}
