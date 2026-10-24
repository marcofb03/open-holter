package org.openholter.recorder

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.RemoteInput

/** "Add comment" (inline reply) and "No comment" on the notification shown after a marker. */
class MarkerCommentReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val markerNs = intent.getLongExtra(RecorderService.EXTRA_MARKER_NS, 0L)
        if (intent.action == ACTION_COMMENT && markerNs > 0) {
            val text = RemoteInput.getResultsFromIntent(intent)?.getCharSequence(KEY_COMMENT)?.toString().orEmpty()
            if (text.isNotBlank()) {
                runCatching { HolterApp.from(context).markers.setComment(markerNs, text) }
                    .onFailure { Log.e("MarkerComment", "could not save comment", it) }
            }
        }
        // An inline reply must be answered by updating or removing the notification.
        NotificationManagerCompat.from(context).cancel(RecorderService.COMMENT_NOTIFICATION_ID)
    }

    companion object {
        const val ACTION_COMMENT = "org.openholter.recorder.MARKER_COMMENT"
        const val ACTION_SKIP = "org.openholter.recorder.MARKER_SKIP"
        const val KEY_COMMENT = "comment"
    }
}
