const docx = require('docx');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, PageBreak, TabStopType, TabStopPosition,
  VerticalAlign, ImageRun
} = docx;

// Fallback for TabLeader which might be named differently or missing in some versions
const TabLeader = docx.TabLeader || docx.TabStopLeader || { DOT: "dot" };
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function p(text, opts = {}) {
  return new Paragraph({
    ...opts,
    children: [new TextRun({ text, font: "Times New Roman", size: 24, ...opts.run })]
  });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: "Times New Roman", size: 28, bold: true })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Times New Roman", size: 26, bold: true })]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: "Times New Roman", size: 24, bold: true })]
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 120, line: 360, lineRule: "auto" },
    indent: { firstLine: 720 },
    children: [new TextRun({ text, font: "Times New Roman", size: 24, ...opts })]
  });
}

function bodyNoIndent(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 120, line: 360, lineRule: "auto" },
    children: [new TextRun({ text, font: "Times New Roman", size: 24, ...opts })]
  });
}

function emptyLine() {
  return new Paragraph({ children: [new TextRun("")] });
}

function centeredBold(text, size = 24) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 120 },
    children: [new TextRun({ text, font: "Times New Roman", size, bold: true })]
  });
}

function centered(text, size = 24) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 120 },
    children: [new TextRun({ text, font: "Times New Roman", size })]
  });
}

// Numbered list item
function numItem(text, ref = "numbers") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 80, line: 360, lineRule: "auto" },
    children: [new TextRun({ text, font: "Times New Roman", size: 24 })]
  });
}

// Letter list item (a, b, c)
function letterItem(text) {
  return new Paragraph({
    numbering: { reference: "letters", level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 80, line: 360, lineRule: "auto" },
    children: [new TextRun({ text, font: "Times New Roman", size: 24 })]
  });
}

function bulletItem(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text, font: "Times New Roman", size: 24 })]
  });
}

function tocRow(text, page, indentLevel = 0, isBold = false) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    indent: { left: indentLevel * 400 },
    tabStops: [
      { type: TabStopType.RIGHT, position: 9000, leader: TabLeader.DOT }
    ],
    children: [
      new TextRun({ text, font: "Times New Roman", size: 24, bold: isBold }),
      new TextRun({ text: "\t", font: "Times New Roman", size: 24, bold: isBold }),
      new TextRun({ text: page, font: "Times New Roman", size: 24, bold: isBold }),
    ],
  });
}

function tocChapter(num, title, page) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    tabStops: [
      { type: TabStopType.LEFT, position: 500 },
      { type: TabStopType.RIGHT, position: 9000, leader: TabLeader.DOT }
    ],
    children: [
      new TextRun({ text: num + "\t" + title, font: "Times New Roman", size: 24, bold: true }),
      new TextRun({ text: "\t", font: "Times New Roman", size: 24, bold: true }),
      new TextRun({ text: page, font: "Times New Roman", size: 24, bold: true }),
    ],
  });
}

const doc = new Document({
  numbering: {
    config: [
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "letters",
        levels: [{
          level: 0, format: LevelFormat.LOWER_LETTER, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 }
      }
    ]
  },
  sections: [
    // ==================== COVER PAGE ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      children: [
        emptyLine(),
        centeredBold("OPTIMASI DINAMIS PEMODELAN NETWORK MARKOWITZ", 28),
        centeredBold("UNTUK MANAJEMEN PORTOFOLIO MATA UANG KRIPTO", 28),
        emptyLine(),
        emptyLine(),
        emptyLine(),
        centered("[Logo Universitas Nusa Mandiri]", 20),
        emptyLine(),
        emptyLine(),
        centeredBold("PROPOSAL TESIS", 26),
        emptyLine(),
        centered("Diajukan sebagai salah satu syarat untuk memperoleh gelar"),
        centered("Magister Komputer (M.Kom)"),
        emptyLine(),
        emptyLine(),
        centeredBold("Ragil Yulianto", 24),
        centered("14240007"),
        emptyLine(),
        emptyLine(),
        centered("Program Studi Ilmu Komputer (S2)"),
        centered("Fakultas Teknologi Informasi"),
        centered("Universitas Nusa Mandiri"),
        centered("Jakarta"),
        centered("2026"),
      ]
    },
    // ==================== HALAMAN PENGESAHAN ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      children: [
        emptyLine(),
        centeredBold("HALAMAN PENGESAHAN", 26),
        emptyLine(),
        centeredBold("PROPOSAL TESIS", 24),
        emptyLine(),
        emptyLine(),
        bodyNoIndent("Proposal tesis ini diajukan oleh:", { bold: true }),
        emptyLine(),
        // Detail section with bold labels and content
        new Paragraph({
          children: [
            new TextRun({ text: "Nama", font: "Times New Roman", size: 24, bold: true }),
            new TextRun({ text: "\t: Ragil Yulianto", font: "Times New Roman", size: 24, bold: true }),
          ],
          tabStops: [{ type: TabStopType.LEFT, position: 2000 }]
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "NIM", font: "Times New Roman", size: 24, bold: true }),
            new TextRun({ text: "\t: 14240007", font: "Times New Roman", size: 24, bold: true }),
          ],
          tabStops: [{ type: TabStopType.LEFT, position: 2000 }]
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "Program Studi", font: "Times New Roman", size: 24, bold: true }),
            new TextRun({ text: "\t: Ilmu Komputer", font: "Times New Roman", size: 24, bold: true }),
          ],
          tabStops: [{ type: TabStopType.LEFT, position: 2000 }]
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "Fakultas", font: "Times New Roman", size: 24, bold: true }),
            new TextRun({ text: "\t: Teknologi Informasi", font: "Times New Roman", size: 24, bold: true }),
          ],
          tabStops: [{ type: TabStopType.LEFT, position: 2000 }]
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "Jenjang", font: "Times New Roman", size: 24, bold: true }),
            new TextRun({ text: "\t: Strata Dua (S2)", font: "Times New Roman", size: 24, bold: true }),
          ],
          tabStops: [{ type: TabStopType.LEFT, position: 2000 }]
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "Judul Tesis", font: "Times New Roman", size: 24, bold: true }),
            new TextRun({ text: "\t: OPTIMASI DINAMIS PEMODELAN NETWORK MAR-", font: "Times New Roman", size: 24, bold: true }),
          ],
          tabStops: [{ type: TabStopType.LEFT, position: 2000 }]
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "\t  KOWITZ UNTUK MANAJEMEN PORTOFOLIO MATA", font: "Times New Roman", size: 24, bold: true }),
          ],
          tabStops: [{ type: TabStopType.LEFT, position: 2000 }]
        }),
        new Paragraph({
          children: [
            new TextRun({ text: "\t  UANG KRIPTO", font: "Times New Roman", size: 24, bold: true }),
          ],
          tabStops: [{ type: TabStopType.LEFT, position: 2000 }]
        }),
        emptyLine(),
        emptyLine(),
        bodyNoIndent("Jakarta, 11 Maret 2026"),
        emptyLine(),
        emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Dosen Pembimbing", font: "Times New Roman", size: 24, bold: true })]
        }),
        emptyLine(),
        emptyLine(),
        emptyLine(),
        emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "____________________________________", font: "Times New Roman", size: 24, bold: true }),
          ]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Nama Dosen Pembimbing", font: "Times New Roman", size: 24, bold: true }),
          ]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "NIDN. XXXXXXXXXX", font: "Times New Roman", size: 24, bold: true }),
          ]
        }),
        emptyLine(),
        emptyLine(),
        emptyLine(),
        emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "ii", font: "Times New Roman", size: 24 })]
        }),
      ]
    },
    // ==================== DAFTAR ISI ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      children: [
        emptyLine(),
        centeredBold("DAFTAR ISI", 26),
        emptyLine(),
        emptyLine(),
        tocRow("HALAMAN PENGESAHAN", "ii", 0, true),
        emptyLine(),
        tocChapter("I", "PENDAHULUAN", "1"),
        tocRow("1.1 Latar Belakang", "1", 1),
        tocRow("1.2 Identifikasi Masalah", "1", 1),
        tocRow("1.3 Tujuan Penelitian", "2", 1),
        tocRow("1.4 Ruang Lingkup Penelitian", "2", 1),
        tocRow("1.5 Sistematika Penulisan", "3", 1),
        emptyLine(),
        tocChapter("II", "LANDASAN/KERANGKA PEMIKIRAN", "4"),
        tocRow("2.1 Kerangka Teori", "4", 1),
        tocRow("2.1.1 Modern Portfolio Theory (Markowitz)", "4", 2),
        tocRow("2.1.2 Random Matrix Theory (RMT) dan Kompleksitas Jaringan", "4", 2),
        tocRow("2.1.3 Network Markowitz", "4", 2),
        tocRow("2.1.4 Penelitian Terdahulu", "4", 2),
        emptyLine(),
        tocChapter("III", "METODOLOGI PENELITIAN", "6"),
        tocRow("3.1 Tahapan Penelitian", "6", 1),
        tocRow("3.2 Alat dan Bahan Penelitian", "7", 1),
        tocRow("3.2.1 Perangkat Lunak", "7", 2),
        tocRow("3.3 Dataset", "7", 1),
        tocRow("3.4 Metode/Algoritma yang Digunakan", "7", 1),
        tocRow("3.5 Rencana Jadwal Penelitian", "8", 1),
        emptyLine(),
        tocRow("DAFTAR REFERENSI", "9", 0, true),
        emptyLine(),
        emptyLine(),
        emptyLine(),
        emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "iii", font: "Times New Roman", size: 24 })]
        }),
      ]
    },
    // ==================== BAB I ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      children: [
        centeredBold("BAB I", 26),
        centeredBold("PENDAHULUAN", 26),
        emptyLine(),
        heading2("1.1. Latar Belakang"),
        body("Mata uang kripto (cryptocurrency) telah berkembang menjadi salah satu aset investasi digital yang sangat diminati namun memiliki tingkat volatilitas ekstrem. Dalam manajemen portofolio tradisional, model Mean-Variance dari Markowitz kerap digunakan untuk mengalokasikan aset demi mencapai kombinasi return dan risiko yang optimal. Sayangnya, model klasik ini sangat rentan terhadap noise and estimasi matriks korelasi yang tidak stabil, terutama pada saat gejolak pasar (market crash) seperti fenomena crypto winter [1]."),
        body("Ketika terjadi guncangan pasar, sebagian besar aset kripto cenderung jatuh secara bersamaan, merusak struktur korelasi normal dan menyebabkan portofolio standar mengalami kerugian parah (drawdown yang dalam). Untuk mengatasi tantangan tersebut, penelitian terkini mulai menggabungkan Teori Jaringan Kompleks (Complex Network Theory) ke dalam optimasi portofolio [2]. Penggunaan instrumen seperti Minimum Spanning Tree (MST) dan Eigenvector Centrality terbukti efisien dalam memetakan interaksi antar aset dan menghukum (penalize) aset-aset yang menjadi titik pusat kegagalan sistemik."),
        body("Kendati model Network Markowitz statis menunjukkan proteksi yang lebih relevan dibandingkan Classical Markowitz, penentuan faktor penalti sentralitas \u03b3 (\u03b3) yang kaku kerap menimbulkan masalah di fase pasar yang dinamis (misalnya fase recovery atau bullish). Oleh karena itu, diperlukan pendekatan adaptif yang mengintegrasikan teknik pembersihan sinyal berbasis Random Matrix Theory (RMT) disandingkan dengan optimalisasi Grid Search dinamis (rolling window) agar portofolio dapat membentengi aset di saat crypto winter tanpa mengorbankan rasio upside gain di saat pembalikan arah (bullish)."),
        emptyLine(),
        heading2("1.2. Identifikasi Masalah"),
        body("Berdasarkan latar belakang di atas, dapat diidentifikasi masalah sebagai berikut:"),
        letterItem("Model Classical Markowitz rentan terhadap spurious correlations pada aset kripto yang bervolatilitas sangat tinggi, terutama pada kondisi krisis ekstrem."),
        letterItem("Implementasi Network Markowitz yang telah ada sering kali menggunakan hiperparameter penalti sentralitas (\u03b3) bernilai konstan (statis), yang berpotensi menjadi bumerang saat pasar memasuki rezim recovery atau bullish."),
        letterItem("Kurangnya model sistematis yang secara metodis menyesuaikan struktur jaringan portofolio dengan rezim sikluk guncangan harga terkini menggunakan teknik optimasi rolling window untuk aset-aset kripto utama."),
        emptyLine(),
        heading2("1.3. Tujuan Penelitian"),
        body("Tujuan dari penelitian ini adalah:"),
        letterItem("Menganalisis keandalan metodologi Network Markowitz (dengan integrasi RMT filter dan Eigenvector Centrality) dalam menekan ekstrimitas downside risk dibandingkan pendekatan portofolio naif dan konvensional."),
        letterItem("Merancang dan menguji model Network Markowitz adaptif (Grid Search Optimization) yang mampu melakukan re-kalibrasi dinamis dengan menggunakan paradigma rolling window pada berbagai lanskap pasar (Bearish, Recovery, Stable)."),
        letterItem("Membandingkan performa perlindungan risiko sistemik (VaR) dan asimetri imbal hasil (Rachev Ratio) antara pemodelan baru dengan metode-metode baseline pada reksadana aset kripto."),
        emptyLine(),
        heading2("1.4. Ruang Lingkup Penelitian"),
        body("Ruang lingkup penelitian ini dibatasi pada:"),
        letterItem("Objek penelitian terfokus pada data fluktuasi harga harian dari 10 (sepuluh) aset kripto utama dalam kerangka waktu historis termasuk masa resesi crypto winter (14 September 2017 hingga 17 Oktober 2019)."),
        letterItem("Metode yang dibandingkan secara teknis mencakup Equally Weighted (EW), Classical Markowitz (CM), Glasso Markowitz (GM), Network Markowitz statis (\u03b3 = 0, 1.0, 2.0), serta Optimized Network Markowitz secara dinamis berbasis Grid Search."),
        letterItem("Pengujian (backtesting) dilakukan dalam out-of-sample rolling window (120 observasi ke belakang dengan frekuensi penyesuaian rebalance 7 hari) yang disimulasikan menggunakan transaction cost atau estimasi biaya bursa (0.1%)."),
        emptyLine(),
        heading2("1.5. Sistematika Penulisan"),
        body("Sistematika penulisan proposal tesis ini disusun sebagai berikut:"),
        emptyLine(),
        new Paragraph({
          spacing: { before: 0, after: 80 },
          children: [new TextRun({ text: "BAB I PENDAHULUAN", font: "Times New Roman", size: 24, bold: true })]
        }),
        body("Bab ini membahas latar belakang penelitian, identifikasi masalah, tujuan penelitian, ruang lingkup penelitian, dan sistematika penulisan."),
        new Paragraph({
          spacing: { before: 80, after: 80 },
          children: [new TextRun({ text: "BAB II LANDASAN/KERANGKA PEMIKIRAN", font: "Times New Roman", size: 24, bold: true })]
        }),
        body("Bab ini membahas kerangka teori yang relevan mencakup Teori Portofolio Modern, Complex Network Theory, dan Manajemen Risiko, serta tinjauan pustaka terhadap penelitian terdahulu di bidang Robo-Advisory kripto."),
        new Paragraph({
          spacing: { before: 80, after: 80 },
          children: [new TextRun({ text: "BAB III METODOLOGI PENELITIAN", font: "Times New Roman", size: 24, bold: true })]
        }),
        body("Bab ini menjelaskan alur sistematis eksperimen yang digunakan untuk memproses data instrumen kripto, penyaringan RMT pada matriks korelasi historis, pembangunan MST, fungsi objektif Markowitz modifikasi, dan komputasi skema backtesting."),
      ]
    },
    // ==================== BAB II ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      children: [
        centeredBold("BAB II", 26),
        centeredBold("LANDASAN/KERANGKA PEMIKIRAN", 26),
        emptyLine(),
        heading2("2.1. Kerangka Teori"),
        heading3("2.1.1. Modern Portfolio Theory (Markowitz)"),
        body("Teori Portofolio Modern, yang dipelopori oleh Harry Markowitz, berupaya memaksimalkan imbal hasil yang diharapkan (expected return) pada tingkat risiko (variance) tertentu, atau sebaliknya. Masalah mendasarnya adalah bahwa kovarians dari kumpulan aset finansial sangat sensitif terhadap nilai-nilai ekstrem historis, yang dikenal sebagai Markowitz Curse [4]."),
        emptyLine(),
        heading3("2.1.2. Random Matrix Theory (RMT) dan Kompleksitas Jaringan"),
        body("Teori Random Matrix memungkinkan disaringnya noise dari struktur korelasi dengan membandingkan nilai eigen (eigenvalues) struktur empiris terhadap nilai batas distribusi teoritis Marchenko-Pastur [3]. Ini merupakan vital element sebelum dilakukan visualisasi graf berupa Minimum Spanning Tree (MST)."),
        emptyLine(),
        heading3("2.1.3. Network Markowitz"),
        body("Diperkenalkan baru-baru ini untuk penanganan robo-advisory pada kripto, komponen sentralitas eigenvector ditambahkan sebagai instrumen penalti di dalam penyelesaian optimasi Mean-Variance [1]. Sentralitas mewakili kerentanan sebuah aset mentransmisikan shock pada seluruh jaringan koin di pasar."),
        emptyLine(),
        heading3("2.1.4. Penelitian Terdahulu"),
        body("Beberapa penelitian terdahulu yang relevan dengan penelitian ini antara lain:"),
        emptyLine(),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Tabel II.1. Perbandingan Penelitian Terdahulu", font: "Times New Roman", size: 24, bold: true })]
        }),
        new Table({
          width: { size: 9026, type: WidthType.DXA },
          columnWidths: [700, 2500, 3000, 2826],
          rows: [
            new TableRow({
              tableHeader: true,
              children: [
                new TableCell({
                  borders, width: { size: 700, type: WidthType.DXA },
                  shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  verticalAlign: VerticalAlign.CENTER,
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "No", font: "Times New Roman", size: 22, bold: true })] })]
                }),
                new TableCell({
                  borders, width: { size: 2500, type: WidthType.DXA },
                  shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Penulis / Tahun", font: "Times New Roman", size: 22, bold: true })] })]
                }),
                new TableCell({
                  borders, width: { size: 3000, type: WidthType.DXA },
                  shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Judul", font: "Times New Roman", size: 22, bold: true })] })]
                }),
                new TableCell({
                  borders, width: { size: 2826, type: WidthType.DXA },
                  shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Hasil", font: "Times New Roman", size: 22, bold: true })] })]
                }),
              ]
            }),
            new TableRow({
              children: [
                new TableCell({
                  borders, width: { size: 700, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "1", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 2500, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "Giudici, et al. (2020)", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 3000, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "Network Models to Improve Automated Cryptocurrency Portfolio Management", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 2826, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "Mengusulkan Network Markowitz dan sukses mendemonstrasikan perbaikan struktur dibandingkan Markowitz biasa di era crypto winter.", font: "Times New Roman", size: 22 })] })]
                }),
              ]
            }),
            new TableRow({
              children: [
                new TableCell({
                  borders, width: { size: 700, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "2", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 2500, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "Papenbrock (2011)", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 3000, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "Financial networks and risk management", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 2826, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "Bukti bahwa penggunaan analisis jaringan meningkatkan akurasi risk metrics.", font: "Times New Roman", size: 22 })] })]
                }),
              ]
            }),
            new TableRow({
              children: [
                new TableCell({
                  borders, width: { size: 700, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "3", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 2500, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "Mantegna (1999)", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 3000, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "Hierarchical structure in financial markets", font: "Times New Roman", size: 22 })] })]
                }),
                new TableCell({
                  borders, width: { size: 2826, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: "MST sukses memotret topologi kedekatan dan konektivitas sektor finansial dari pergerakan deret waktu pasar saham.", font: "Times New Roman", size: 22 })] })]
                }),
              ]
            }),
          ]
        }),
      ]
    },
    // ==================== BAB III ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      children: [
        centeredBold("BAB III", 26),
        centeredBold("METODOLOGI PENELITIAN", 26),
        emptyLine(),
        heading2("3.1. Tahapan Penelitian"),
        body("Penelitian ini dilaksanakan melalui tahapan-tahapan sebagai berikut:"),
        numItem("Studi Literatur"),
        body("Melakukan kajian komprehensif terhadap jurnal yang membahas crypto portfolio optimization, graph theory, dan Network Markowitz."),
        numItem("Pengumpulan Data & Pengkondisian"),
        body("Mengumpulkan seri harga close cryptocurrency. Harga lalu diubah menjadi format log returns."),
        numItem("Pra-pemrosesan Data (RMT Filtering)"),
        body("Data returns akan diubah menjadi Matriks Korelasi (Pearson). Selanjutnya struktur noise historis dipotong menggunakan mekanisme filtrasi nilai eigen (eigenvalue clipping boundary batas Marchenko-Pastur)."),
        numItem("Kuantifikasi Jaringan Aset (MST & Centrality)"),
        body("Pembangunan jarak konektivitas (distance matrix) dari hasil korelasi terfilter untuk diekstraksi ke bentuk graf pohon Minimum Spanning Tree (MST). Node importance kemudian diukur melalui Eigenvector Centrality."),
        numItem("Optimasi Jaringan Adaptif (Dynamic Grid-Search)"),
        body("Merancang algoritma komputasi untuk menyesuaikan parameter impact korelasi (\u03b3) yang secara otomatis mengekang instrumen-instrumen bervolatilitas sistemik pada jendela uji terkalibrasi ke belakang (backtrack validity)."),
        numItem("Eksekusi Backtesting Portofolio"),
        body("Mensimulasikan pembelian pada titik waktu (t) dan meninjau portofolio secara periodik menggunakan sistem Rolling Window. Terdapat perlakuan pengenaan slippage/transaction log pada rebalancing harian."),
        numItem("Pengukuran Evaluasi Resiko (Performance Metrics)"),
        body("Mengukur nilai profit kumulatif, asimetri VaR 95%, hingga penyesuaian Sharpe Ratio dan Rachev Ratio di sepanjang berbagai transisi fasa pasar ekstrem."),
        emptyLine(),
        heading2("3.2. Alat dan Bahan Penelitian"),
        heading3("3.2.1. Perangkat Lunak"),
        bulletItem("Sistem Operasi: Windows 10/11"),
        bulletItem("Bahasa Pemrograman: Python 3.x (dengan ekosistem Anaconda)"),
        bulletItem("Framework/Library: Pandas, Numpy, Scipy (Optimization), Scikit-Learn, NetworkX (Graph Analytics)"),
        emptyLine(),
        heading2("3.3. Dataset"),
        body("Dataset yang akan digunakan adalah sekumpulan (pool) 10 aset berbasis cryptocurrency berkapitalisasi tinggi yang tercatat aktif diperdagangkan secara bersinambung. Terdiri dari Bitcoin (BTC), Ethereum (ETH), Ripple (XRP), Litecoin (LTC), dan lain sebagainya. Observasi direntangkan secara sengaja mencakup era penggelembungan (speculative bubble), era krisis berkepanjangan (crypto winter bear market), hingga skema normalisasi stable."),
        emptyLine(),
        heading2("3.4. Metode/Algoritma yang Digunakan"),
        body("Optimasi yang diajukan akan merubah fungsi pencarian model klasik Markowitz menjadi kerangka berbasis penalty function dinamis. Formula Network Markowitz:"),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 120, after: 120 },
          children: [new TextRun({ text: "min\u1D64  w\u1D40 \u00B7 Sf \u00B7 w + \u03B3 \u03A3(Ce \u00B7 w)    ... (III.1)", font: "Times New Roman", size: 24, italics: true })]
        }),
        body("Keterangan:"),
        bodyNoIndent("w  = Vektor alokasi bobot untuk setiap aset kripto (sumbangan = 1)."),
        bodyNoIndent("Sf = Matriks Kovarians terfilter RMT."),
        bodyNoIndent("\u03B3  = Parameter skalar reguler untuk tingkat penghukuman (penalty level) sentralitas graf."),
        bodyNoIndent("Ce = Vektor skor Eigenvector Centrality tiap-tiap entitas node."),
        emptyLine(),
        body("Pada penelitian ini, bobot \u03b3 tidak akan dilakukan hard-coded statis, melainkan secara luwes dan rolling akan difungsikan optimasi grid validation berbasis metrik obyektif Sharpe Ratio periode belakang."),
        emptyLine(),
        heading2("3.5. Rencana Jadwal Penelitian"),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Tabel III.1. Rencana Jadwal Penelitian", font: "Times New Roman", size: 24, bold: true })]
        }),
        new Table({
          width: { size: 9026, type: WidthType.DXA },
          columnWidths: [2900, 1021, 1021, 1021, 1021, 1021, 1021],
          rows: [
            new TableRow({
              tableHeader: true,
              children: [
                new TableCell({
                  borders, width: { size: 2900, type: WidthType.DXA },
                  shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Kegiatan", font: "Times New Roman", size: 22, bold: true })] })]
                }),
                ...["Bln 1", "Bln 2", "Bln 3", "Bln 4", "Bln 5", "Bln 6"].map(h => new TableCell({
                  borders, width: { size: 1021, type: WidthType.DXA },
                  shading: { fill: "D5E8F0", type: ShadingType.CLEAR },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: h, font: "Times New Roman", size: 22, bold: true })] })]
                }))
              ]
            }),
            ...[
              ["Studi Literatur", ["✓", "", "", "", "", ""]],
              ["Pengumpulan Data", ["✓", "✓", "", "", "", ""]],
              ["Pra-pemrosesan Data", ["", "✓", "✓", "", "", ""]],
              ["Perancangan Model/Sistem", ["", "", "✓", "✓", "", ""]],
              ["Implementasi", ["", "", "", "✓", "✓", ""]],
              ["Pengujian dan Evaluasi", ["", "", "", "", "✓", "✓"]],
              ["Penulisan Laporan", ["✓", "✓", "✓", "✓", "", ""]],
            ].map(([activity, marks]) => new TableRow({
              children: [
                new TableCell({
                  borders, width: { size: 2900, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ children: [new TextRun({ text: activity, font: "Times New Roman", size: 22 })] })]
                }),
                ...marks.map(m => new TableCell({
                  borders, width: { size: 1021, type: WidthType.DXA },
                  margins: { top: 80, bottom: 80, left: 120, right: 120 },
                  children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: m, font: "Times New Roman", size: 22 })] })]
                }))
              ]
            }))
          ]
        }),
      ]
    },
    // ==================== DAFTAR REFERENSI ====================
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1800 }
        }
      },
      children: [
        centeredBold("DAFTAR REFERENSI", 26),
        emptyLine(),
        new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { before: 0, after: 120, line: 360 },
          indent: { left: 720, hanging: 720 },
          children: [new TextRun({ text: "[1] P. Giudici, A. Sariev, and G. Toscani, \"Network Models to Improve Automated Cryptocurrency Portfolio Management,\" Risks, vol. 8, no. 3, p. 96, 2020. https://doi.org/10.3390/risks8030096.", font: "Times New Roman", size: 24 })]
        }),
        new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { before: 0, after: 120, line: 360 },
          indent: { left: 720, hanging: 720 },
          children: [new TextRun({ text: "[2] S. S. Momeni dan R. E. Rostami, \"Portfolio selection using a hybrid methodology of network theory and optimization,\" Annals of Operations Research, vol. 308, pp. 317–345, 2021.", font: "Times New Roman", size: 24 })]
        }),
        new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { before: 0, after: 120, line: 360 },
          indent: { left: 720, hanging: 720 },
          children: [new TextRun({ text: "[3] V. A. Marchenko and L. A. Pastur, \"Distribution of eigenvalues for some sets of random matrices,\" Mathematics of the USSR-Sbornik, vol. 1, no. 4, pp. 457–483, 1967.", font: "Times New Roman", size: 24 })]
        }),
        new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { before: 0, after: 120, line: 360 },
          indent: { left: 720, hanging: 720 },
          children: [new TextRun({ text: "[4] H. Markowitz, \"Portfolio Selection,\" The Journal of Finance, vol. 7, no. 1, pp. 77–91, 1952.", font: "Times New Roman", size: 24 })]
        }),
        new Paragraph({
          alignment: AlignmentType.JUSTIFIED,
          spacing: { before: 0, after: 120, line: 360 },
          indent: { left: 720, hanging: 720 },
          children: [new TextRun({ text: "[5] R. N. Mantegna, \"Hierarchical structure in financial markets,\" The European Physical Journal B, vol. 11, no. 1, pp. 193–197, 1999.", font: "Times New Roman", size: 24 })]
        }),
      ]
    },
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("proposal_tesis_ragil.docx", buffer);
  console.log("Done!");
});
