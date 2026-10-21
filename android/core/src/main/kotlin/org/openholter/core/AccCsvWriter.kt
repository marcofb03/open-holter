package org.openholter.core

import java.io.Closeable
import java.io.File
import java.time.ZoneId

/**
 * H10 accelerometer samples in milli-g, hourly files next to the ECG files:
 * `timestamp_ns,x_mG,y_mG,z_mG`. Lets the analysis tell movement artefacts
 * from real beats.
 */
class AccCsvWriter(
    root: File,
    zone: ZoneId,
    flushIntervalNs: Long = 10_000_000_000L,
    syncIntervalNs: Long = 60_000_000_000L,
) : Closeable {

    private val file = HourlyFileWriter(
        zone, RecordingFiles.ACC_HEADER, { RecordingFiles.accFile(root, it) },
        flushIntervalNs, syncIntervalNs,
    )
    private val row = StringBuilder(48)

    fun write(timestampNs: Long, xMg: Int, yMg: Int, zMg: Int) {
        row.setLength(0)
        row.append(timestampNs).append(',').append(xMg).append(',').append(yMg).append(',').append(zMg)
        file.writeLine(timestampNs, row)
    }

    fun tick(nowNs: Long) = file.tick(nowNs)

    fun flush(sync: Boolean = false) = file.flush(sync)

    override fun close() = file.close()
}
