package com.hemascan.ai

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class Cell(val x: Float, val y: Float, val r: Float, val label: String, val conf: Float)

object Analyzer {
    const val UNCLASSIFIED = "Cellule"
    const val UNCERTAIN = "Incertaine"
    const val WBC = "Leucocyte (suggestion)"
    const val PLT = "Plaquette (suggestion)"

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

    private fun localMax(dist: Mat, k: Int, minR: Double): Mat {
        val dil = Mat()
        Imgproc.dilate(dist, dil, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble())))
        val eq = Mat(); Core.compare(dist, dil, eq, Core.CMP_EQ)
        val above = Mat(); Core.compare(dist, Scalar(minR), above, Core.CMP_GT)
        Core.bitwise_and(eq, above, eq)
        return eq
    }

    private fun comps(peaks: Mat, dist: Mat): List<DoubleArray> {
        val lab = Mat(); val st = Mat(); val ce = Mat()
        val n = Imgproc.connectedComponentsWithStats(peaks, lab, st, ce)
        val res = ArrayList<DoubleArray>()
        for (i in 1 until n) {
            val x = ce.get(i, 0)[0]; val y = ce.get(i, 1)[0]
            val r = dist.get(y.toInt().coerceIn(0, dist.rows() - 1), x.toInt().coerceIn(0, dist.cols() - 1))[0]
            res.add(doubleArrayOf(x, y, r))
        }
        return res
    }

    /** sens : 0.5 (strict) à 1.5 (sensible). */
    fun detect(src: Bitmap, sens: Float = 1f): List<Cell> {
        val f = min(1f, 1200f / max(src.width, src.height))
        val bmp = if (f < 1f) Bitmap.createScaledBitmap(src, (src.width * f).toInt(), (src.height * f).toInt(), true) else src
        val rgba = Mat(); Utils.bitmapToMat(bmp, rgba)
        val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        val gray = Mat(); Imgproc.cvtColor(rgb, gray, Imgproc.COLOR_RGB2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(7.0, 7.0), 0.0)
        val w = gray.cols(); val h = gray.rows()

        // 1. Champ circulaire du microscope (on ignore le noir autour)
        val pw = max(4, w / 20); val ph = max(4, h / 20)
        val corner = (Core.mean(gray.submat(0, ph, 0, pw)).`val`[0] + Core.mean(gray.submat(0, ph, w - pw, w)).`val`[0] +
            Core.mean(gray.submat(h - ph, h, 0, pw)).`val`[0] + Core.mean(gray.submat(h - ph, h, w - pw, w)).`val`[0]) / 4
        val center = Core.mean(gray.submat(h * 2 / 5, h * 3 / 5, w * 2 / 5, w * 3 / 5)).`val`[0]
        val mask = Mat.zeros(h, w, CvType.CV_8UC1)
        var circle = false; var cx = w / 2.0; var cy = h / 2.0; var fr = 1e9
        if (corner < 0.5 * center && center > 60) {
            val m = Mat(); Imgproc.threshold(gray, m, max(40.0, 0.45 * center), 255.0, Imgproc.THRESH_BINARY)
            val mo = Imgproc.moments(m, true)
            if (mo.m00 > 100) { circle = true; cx = mo.m10 / mo.m00; cy = mo.m01 / mo.m00; fr = sqrt(mo.m00 / PI) }
        }
        if (circle) Imgproc.circle(mask, Point(cx, cy), (fr * 0.97).toInt(), Scalar(255.0), -1)
        else mask.setTo(Scalar(255.0))

        // 2. Carte de différence de couleur avec le fond (Lab)
        val lab = Mat(); Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val labs = ArrayList<Mat>(); Core.split(lab, labs)
        val ms = MatOfDouble(); val sd = MatOfDouble()
        Core.meanStdDev(labs[0], ms, sd, mask)
        val bright = Mat()
        Imgproc.threshold(labs[0], bright, ms.get(0, 0)[0] + 0.2 * sd.get(0, 0)[0], 255.0, Imgproc.THRESH_BINARY)
        Core.bitwise_and(bright, mask, bright)
        val bg = if (Core.countNonZero(bright) > 50) Core.mean(lab, bright) else Core.mean(lab, mask)
        val lab32 = Mat(); lab.convertTo(lab32, CvType.CV_32FC3)
        Core.subtract(lab32, Scalar(bg.`val`[0], bg.`val`[1], bg.`val`[2], 0.0), lab32)
        Core.multiply(lab32, lab32, lab32)
        val chs = ArrayList<Mat>(); Core.split(lab32, chs)
        val sum = Mat(); Core.add(chs[0], chs[1], sum); Core.add(sum, chs[2], sum)
        Core.sqrt(sum, sum)
        val d8 = Mat(); sum.convertTo(d8, CvType.CV_8U, 255.0 / 100.0)

        // 3. Seuillage
        val masked = Mat(); d8.copyTo(masked, mask)
        val tmp = Mat()
        val otsu = Imgproc.threshold(masked, tmp, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        val bin = Mat()
        Imgproc.threshold(masked, bin, (otsu / sens).coerceIn(8.0, 90.0), 255.0, Imgproc.THRESH_BINARY)
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0)))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(7.0, 7.0)))
        val cts = ArrayList<MatOfPoint>()
        Imgproc.findContours(bin.clone(), cts, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val filled = Mat.zeros(h, w, CvType.CV_8UC1)
        val big = cts.filter { Imgproc.contourArea(it) >= 50 }
        if (big.isEmpty()) return emptyList()
        Imgproc.drawContours(filled, big, -1, Scalar(255.0), -1)

        // 4. Séparation des cellules collées (maxima locaux de la carte de distance)
        val dist = Mat(); Imgproc.distanceTransform(filled, dist, Imgproc.DIST_L2, 5)
        val p1 = comps(localMax(dist, 7, 3.0), dist)
        if (p1.isEmpty()) return emptyList()
        val rs = p1.map { it[2] }.sorted()
        val rTyp = rs[(rs.size * 0.75).toInt().coerceAtMost(rs.size - 1)]
        val k = (2 * (0.7 * rTyp).toInt() + 1).coerceAtLeast(5)
        val raw = comps(localMax(dist, k, max(3.0, 0.35 * rTyp)), dist).sortedByDescending { it[2] }.take(2500)
        if (raw.isEmpty()) return emptyList()

        // 5. Caractéristiques et suggestions (taille + intensité de coloration)
        val dens = raw.map { c ->
            val r = max(2, (c[2] * 0.6).toInt()); val x = c[0].toInt(); val y = c[1].toInt()
            val x0 = max(0, x - r); val x1 = min(w, x + r + 1); val y0 = max(0, y - r); val y1 = min(h, y + r + 1)
            if (x1 <= x0 || y1 <= y0) 0.0 else Core.mean(d8.submat(y0, y1, x0, x1)).`val`[0]
        }
        val medR = raw.map { it[2] }.sorted()[raw.size / 2]
        val medD = dens.sorted()[raw.size / 2]
        return raw.mapIndexed { i, c ->
            val r = c[2]
            val edge = if (circle) hypot(c[0] - cx, c[1] - cy) + r > fr * 0.95
            else c[0] < r || c[1] < r || c[0] > w - r || c[1] > h - r
            val (label, conf) = when {
                edge || r > 1.8 * medR -> Pair(UNCERTAIN, 0f)
                r > 1.25 * medR && dens[i] > 1.4 * medD -> Pair(WBC, 0.4f)
                r < 0.5 * medR -> Pair(PLT, 0.3f)
                else -> Pair(UNCLASSIFIED, 0f)
            }
            Cell((c[0] / f).toFloat(), (c[1] / f).toFloat(), (r / f).toFloat(), label, conf)
        }
    }
}
