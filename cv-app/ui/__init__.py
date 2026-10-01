"""Interfaces utilisateur.

* :mod:`ui.streamlit_app` — interface web (téléversement, webcam par
  instantané, chaîne de filtres, mesures, catalogue) ;
* :mod:`ui.desktop` — fenêtre OpenCV avec trackbars et flux webcam temps réel,
  dans la continuité directe du TP3 ;
* :mod:`ui.widgets` — fabrique de widgets Streamlit à partir des
  :class:`~cvlab.params.ParamSpec`.

Ces modules dépendent de :mod:`cvlab`, jamais l'inverse.
"""
