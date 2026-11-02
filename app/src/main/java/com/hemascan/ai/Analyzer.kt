package com.hemascan.ai

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

data class Cell(val x: Float, val y: Float, val r: Float, val label: String, val conf: Float)

object Analyzer {
    const val UNCLASSIFIED = "Cellule"
    const val UNCERTAIN = "Incertaine"
    const val WBC = "Leucocyte (suggestion)"
    const val PLT = "Plaquette (suggestion)"

    private fun ell(k: Int) = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))

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

    private fun otsu(c8: Mat, mask: Mat): Double {
        val hist = Mat()
        Imgproc.calcHist(listOf(c8), MatOfInt(0), mask, hist, MatOfInt(256), MatOfFloat(0f, 256f))
        val h = DoubleArray(256) { hist.get(it, 0)[0] }
        val total = h.sum()
        var sumAll = 0.0
        for (i in 0..255) sumAll += i * h[i]
        var wB = 0.0; var sumB = 0.0; var best = 0.0; var thr = 0.0
        for (t in 0..255) {
            wB += h[t]
            if (wB == 0.0) continue
            val wF = total - wB
            if (wF == 0.0) break
            sumB += t * h[t]
            val mB = sumB / wB
            val mF = (sumAll - sumB) / wF
            val v = wB * wF * (mB - mF) * (mB - mF)
            if (v > best) { best = v; thr = t.toDouble() }
        }
        return thr
    }

    private fun localMax(dist: Mat, k: Int, minR: Double): Mat {
        val dil = Mat()
        Imgproc.dilate(dist, dil, ell(k))
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
    fun detect(src: Bitmap, sens: Float = 1f): List<Cell> = try {
        detectImpl(src, sens)
    } catch (e: Throwable) {
        listOf(Cell(40f, 40f, 30f, "ERREUR: ${e.javaClass.simpleName} ${e.message}".take(90), 0f))
    }

    private fun detectImpl(src: Bitmap, sens: Float): List<Cell> {
        val f = min(1f, 1200f / max(src.width, src.height))
        val bmp = if (f < 1f) Bitmap.createScaledBitmap(src, (src.width * f).toInt(), (src.height * f).toInt(), true) else src
        val rgba = Mat(); Utils.bitmapToMat(bmp, rgba)
        val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        val w = rgb.cols(); val h = rgb.rows()

        // 1. Carte de "chroma" Lab : le fond noir/gris du microscope a ~0, le rose ~20, les cellules violettes 50+
        val blur = Mat(); Imgproc.GaussianBlur(rgb, blur, Size(5.0, 5.0), 0.0)
        val lab = Mat(); Imgproc.cvtColor(blur, lab, Imgproc.COLOR_RGB2Lab)
        val ch = ArrayList<Mat>(); Core.split(lab, ch)
        val a = Mat(); val b = Mat()
        ch[1].convertTo(a, CvType.CV_32F, 1.0, -128.0)
        ch[2].convertTo(b, CvType.CV_32F, 1.0, -128.0)
        val chroma = Mat(); Core.magnitude(a, b, chroma)

        // 2. Champ du microscope = plus grande zone colorée, remplie
        val fld32 = Mat(); Imgproc.threshold(chroma, fld32, 7.0, 255.0, Imgproc.THRESH_BINARY)
        val fld = Mat(); fld32.convertTo(fld, CvType.CV_8U)
        Imgproc.morphologyEx(fld, fld, Imgproc.MORPH_CLOSE, ell(15))
        val lbl = Mat(); val st = Mat(); val ce = Mat()
        val n = Imgproc.connectedComponentsWithStats(fld, lbl, st, ce)
        if (n < 2) return emptyList()
        var bi = 1; var ba = 0.0
        for (i in 1 until n) {
            val ar = st.get(i, Imgproc.CC_STAT_AREA)[0]
            if (ar > ba) { ba = ar; bi = i }
        }
        val bigMask = Mat(); Core.compare(lbl, Scalar(bi.toDouble()), bigMask, Core.CMP_EQ)
        val fc = ArrayList<MatOfPoint>()
        Imgproc.findContours(bigMask.clone(), fc, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val field = Mat.zeros(h, w, CvType.CV_8UC1)
        Imgproc.drawContours(field, fc, -1, Scalar(255.0), -1)
        Imgproc.erode(field, field, ell(9))
        val fdist = Mat(); Imgproc.distanceTransform(field, fdist, Imgproc.DIST_L2, 5)

        // 3. Seuil d'Otsu sur la chroma, dans le champ uniquement
        val c8 = Mat(); chroma.convertTo(c8, CvType.CV_8U, 3.0)
        val t = (otsu(c8, field) / sens).coerceIn(10.0, 240.0)
        val bin = Mat(); Imgproc.threshold(c8, bin, t, 255.0, Imgproc.THRESH_BINARY)
        Core.bitwise_and(bin, field, bin)
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, ell(3))
        val cts = ArrayList<MatOfPoint>()
        Imgproc.findContours(bin.clone(), cts, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        val keep = cts.filter { Imgproc.contourArea(it) >= 40 }
        if (keep.isEmpty()) return emptyList()
        val filled = Mat.zeros(h, w, CvType.CV_8UC1)
        Imgproc.drawContours(filled, keep, -1, Scalar(255.0), -1)

        // 4. Séparation des cellules collées
        val dist = Mat(); Imgproc.distanceTransform(filled, dist, Imgproc.DIST_L2, 5)
        val p1 = comps(localMax(dist, 7, 3.0), dist)
        if (p1.isEmpty()) return emptyList()
        val rs = p1.map { it[2] }.sorted()
        val rTyp = rs[(rs.size * 0.75).toInt().coerceAtMost(rs.size - 1)]
        val k = (2 * (0.7 * rTyp).toInt() + 1).coerceAtLeast(5)
        val raw = comps(localMax(dist, k, max(3.0, 0.35 * rTyp)), dist).sortedByDescending { it[2] }.take(2500)
        if (raw.isEmpty()) return emptyList()

        // 5. Suggestions simples (taille + intensité) et cellules incertaines
        val dens = raw.map { c ->
            val r = max(2, (c[2] * 0.6).toInt()); val x = c[0].toInt(); val y = c[1].toInt()
            val x0 = max(0, x - r); val x1 = min(w, x + r + 1); val y0 = max(0, y - r); val y1 = min(h, y + r + 1)
            if (x1 <= x0 || y1 <= y0) 0.0 else Core.mean(c8.submat(y0, y1, x0, x1)).`val`[0]
        }
        val medR = raw.map { it[2] }.sorted()[raw.size / 2]
        val medD = dens.sorted()[raw.size / 2]
        return raw.mapIndexed { i, c ->
            val r = c[2]
            val xi = c[0].toInt().coerceIn(0, w - 1); val yi = c[1].toInt().coerceIn(0, h - 1)
            val edge = fdist.get(yi, xi)[0] < 0.9 * r ||
                c[0] < r || c[1] < r || c[0] > w - r || c[1] > h - r
            val (label, conf) = when {
                edge || r > 1.8 * medR -> Pair(UNCERTAIN, 0f)
                r > 1.25 * medR && dens[i] > 1.25 * medD -> Pair(WBC, 0.4f)
                r < 0.5 * medR -> Pair(PLT, 0.3f)
                else -> Pair(UNCLASSIFIED, 0f)
            }
            Cell((c[0] / f).toFloat(), (c[1] / f).toFloat(), (r / f).toFloat(), label, conf)
        }
    }
}
