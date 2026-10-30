package com.hemascan.ai

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

data class Cell(val x: Float, val y: Float, val r: Float, val label: String, val conf: Float)

object Analyzer {
    const val UNCLASSIFIED = "Non classée"
    const val UNCERTAIN = "Incertaine"

    fun enhance(src: Bitmap): Bitmap {
        val m = Mat(); Utils.bitmapToMat(src, m)
        val rgb = Mat(); Imgproc.cvtColor(m, rgb, Imgproc.COLOR_RGBA2RGB)
        val lab = Mat(); Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val ch = ArrayList<Mat>(); Core.split(lab, ch)
        Imgproc.createCLAHE(2.0, Size(8.0, 8.0)).apply(ch[0], ch[0])
        Core.merge(ch, lab)
        Imgproc.cvtColor(lab, rgb, Imgproc.COLOR_Lab2RGB)
        val blur = Mat(); Imgproc.GaussianBlur(rgb, blur, Size(0.0, 0.0), 3.0)
        Core.addWeighted(rgb, 1.5, blur, -0.5, 0.0, rgb)
        val out = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgb, out)
        return out
    }

    fun detect(src: Bitmap): List<Cell> {
        val rgba = Mat(); Utils.bitmapToMat(src, rgba)
        val gray = Mat(); Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        val bin = Mat()
        Imgproc.threshold(gray, bin, 0.0, 255.0, Imgproc.THRESH_BINARY_INV or Imgproc.THRESH_OTSU)
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, k)
        val dist = Mat(); Imgproc.distanceTransform(bin, dist, Imgproc.DIST_L2, 5)
        val mm = Core.minMaxLoc(dist)
        if (mm.maxVal < 6) return emptyList()
        val peaks = Mat(); Imgproc.threshold(dist, peaks, 0.55 * mm.maxVal, 255.0, Imgproc.THRESH_BINARY)
        peaks.convertTo(peaks, CvType.CV_8U)
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(peaks, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val raw = ArrayList<FloatArray>()
        for (c in contours) {
            val rect = Imgproc.boundingRect(c)
            val sub = dist.submat(rect)
            val r = Core.minMaxLoc(sub).maxVal.toFloat()
            if (r < 6f) continue
            raw.add(floatArrayOf(rect.x + rect.width / 2f, rect.y + rect.height / 2f, r))
        }
        if (raw.isEmpty()) return emptyList()
        val med = raw.map { it[2] }.sorted()[raw.size / 2]
        return raw.map {
            val edge = it[0] < it[2] || it[1] < it[2] || it[0] > src.width - it[2] || it[1] > src.height - it[2]
            val odd = it[2] < 0.6f * med || it[2] > 1.6f * med
            Cell(it[0], it[1], it[2], if (edge || odd) UNCERTAIN else UNCLASSIFIED, 0f)
        }
    }
}
