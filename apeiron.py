import sys
from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from ui.main_window import MandelbrotViewer


def main():
    # Ensure consistent decimal separators across locales
    QLocale.setDefault(QLocale(QLocale.C))

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 12))

    viewer = MandelbrotViewer()
    viewer.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
