package org.openholter.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset

class SideFilesTest {
    @get:Rule val tmp = TemporaryFolder()

    private val zone = ZoneOffset.UTC
    private val hour = LocalDateTime.of(2026, 10, 21, 9, 0)
    private val t0 = hour.toEpochSecond(ZoneOffset.UTC) * 1_000_000_000L

    @Test
    fun `accelerometer file layout`() {
        val w = AccCsvWriter(tmp.root, zone)
        w.write(t0, -12, 980, 31)
        w.close()
        assertEquals(
            listOf("timestamp_ns,x_mG,y_mG,z_mG", "$t0,-12,980,31"),
            RecordingFiles.accFile(tmp.root, hour).readLines(),
        )
        assertEquals("acc_2026-10-21_09.csv", RecordingFiles.accFile(tmp.root, hour).name)
    }

    @Test
    fun `heart rate file has one row per RR and contact flags`() {
        val w = HrCsvWriter(tmp.root, zone)
        w.write(t0, 62, listOf(968, 975), contact = true, contactSupported = true)
        w.write(t0 + 1_000_000_000L, 63, emptyList(), contact = false, contactSupported = true)
        w.close()
        assertEquals(
            listOf(
                "timestamp_ns,hr_bpm,rr_ms,contact,contact_supported",
                "$t0,62,968,1,1",
                "$t0,62,975,1,1",
                "${t0 + 1_000_000_000L},63,,0,1",
            ),
            RecordingFiles.hrFile(tmp.root, hour).readLines(),
        )
    }

    @Test
    fun `event log is tab separated with header`() {
        val log = EventLog(tmp.root, zone)
        log.log(t0, "connected", "A1B2C3D4")
        log.log(t0 + 1_500_000_000L, "disconnected", "tab\there")
        log.log(t0 + 2_000_000_000L, "h10_info", "firmware 5.0.0\u0000")
        val lines = RecordingFiles.eventsFile(tmp.root, LocalDate.of(2026, 10, 21)).readLines()
        assertEquals(RecordingFiles.EVENTS_HEADER, lines[0])
        assertEquals("2026-10-21 09:00:00.000\t$t0\tconnected\tA1B2C3D4", lines[1])
        assertEquals("2026-10-21 09:00:01.500\t${t0 + 1_500_000_000L}\tdisconnected\ttab here", lines[2])
        assertEquals("2026-10-21 09:00:02.000\t${t0 + 2_000_000_000L}\th10_info\tfirmware 5.0.0", lines[3])
    }

    @Test
    fun `lists day folders newest first`() {
        tmp.newFolder("2026-10-20")
        tmp.newFolder("2026-10-21")
        tmp.newFolder("not-a-day")
        assertEquals(
            listOf(LocalDate.of(2026, 10, 21), LocalDate.of(2026, 10, 20)),
            RecordingFiles.listDays(tmp.root),
        )
    }

    @Test
    fun `preview buffer keeps the newest values in order`() {
        val b = EcgPreviewBuffer(3)
        assertEquals(0, b.snapshot().size)
        b.add(1f); b.add(2f)
        assertTrue(floatArrayOf(1f, 2f).contentEquals(b.snapshot()))
        b.add(3f); b.add(4f)
        assertTrue(floatArrayOf(2f, 3f, 4f).contentEquals(b.snapshot()))
    }
}
