"""Tests de la normalisation du texte avant synthèse.

Deux régressions déjà rencontrées sont explicitement verrouillées ici : la suppression
des virgules de coupure et celle des deux-points de fin. Toutes deux laissaient des
fragments sans aucune ponctuation finale, ce qui pousse le modèle à prononcer « point ».
"""

from __future__ import annotations

from voxlibris import normalize as N


class TestPonctuationMuette:
    def test_guillemets_supprimes(self):
        assert "«" not in N.normalize("Il dit : « Bonjour »")
        assert "»" not in N.normalize("Il dit : « Bonjour »")

    def test_parentheses_devenues_virgules(self):
        # Supprimées sèchement, l'incise se fondrait dans la phrase porteuse.
        out = N.normalize("La tour fut bâtie (nul ne le sait plus) en 1650.")
        assert "(" not in out and ")" not in out
        assert "bâtie, nul ne le sait plus, en" in out

    def test_tiret_de_dialogue_retire_en_tete(self):
        assert N.normalize("— Bonjour, dit-il.").startswith("Bonjour")

    def test_tiret_en_incise_devenu_virgule(self):
        out = N.normalize("Le phare veillait seul sur la côte — nul ne s'en souvenait.")
        assert "—" not in out
        assert "côte, nul ne s'en souvenait." in out


class TestPonctuationConservee:
    """Régressions : ces ponctuations finales étaient mangées."""

    def test_deux_points_final_conserve(self):
        # « L'aîné dit : » perdait son deux-points et finissait sans ponctuation.
        assert N.normalize("L'aîné dit :").endswith(":")

    def test_virgule_de_coupure_conservee(self):
        # Quand une phrase trop longue est coupée, le fragment doit garder sa virgule.
        texte, _ = N.defuse_ellipsis("Seuls les enfants du port, indifférents au vacarme,")
        assert texte.endswith(",")

    def test_ponctuation_faible_de_tete_retiree(self):
        texte, _ = N.defuse_ellipsis(", puis la marée se retira lentement.")
        assert texte.startswith("puis")


class TestPointsDeSuspension:
    """Les suspensions deviennent du silence, pas du texte."""

    def test_suspension_finale_convertie_en_pause(self):
        texte, pause = N.defuse_ellipsis("Je me demandais...")
        assert texte == "Je me demandais"
        assert pause == N.PAUSE_ELLIPSIS

    def test_suspension_mediane_devenue_virgule(self):
        texte, pause = N.defuse_ellipsis("Ah! merci... enfin... bref, murmura le gardien.")
        assert "..." not in texte
        assert texte == "Ah! merci, enfin, bref, murmura le gardien."
        assert pause == 0

    def test_suspension_initiale_supprimee(self):
        texte, _ = N.defuse_ellipsis("... si vous vouliez bien patienter.")
        assert texte.startswith("si vous")


class TestNombresEtCapitales:
    def test_nombre_ecrit_en_lettres(self):
        out = N.normalize("La tour fut bâtie en 1650 et précédait le port.")
        assert "mille six cent cinquante" in out
        # Le nombre ne doit pas se coller au mot suivant.
        assert "cinquante et" in out

    def test_aucun_chiffre_ne_survit(self):
        out = N.normalize("En l'an 2000, trois cent cinquante ans après.")
        assert not any(c.isdigit() for c in out)

    def test_capitales_isolees_capitalisees(self):
        # Un mot tout en majuscules serait épelé lettre par lettre par le modèle.
        assert "Terre" in N.normalize("— TERRE !")


class TestDecoupe:
    def test_segments_sous_la_limite(self):
        paragraphe = " ".join(["Une phrase de longueur raisonnable."] * 40)
        for segment, _ in N.segment_paragraph(N.normalize(paragraphe)):
            assert len(segment) <= N.MAX_CHARS

    def test_phrase_courte_non_decoupee(self):
        texte = "Une phrase unique et brève."
        assert N.segment_paragraph(texte) == [(texte, True)]

    def test_coupure_preferee_a_la_ponctuation_faible(self):
        # Une phrase trop longue se coupe sur ses virgules, pas au milieu des mots.
        longue = ", ".join(["un membre de phrase assez long pour compter"] * 8) + "."
        segments = N.segment_paragraph(longue)
        assert len(segments) > 1
        assert all(len(s) <= N.MAX_CHARS for s, _ in segments)
        assert all(not s.endswith(" ") for s, _ in segments)
        # Seul le dernier fragment achève la phrase : les autres n'ont pas à recevoir
        # le silence d'un point qu'ils ne portent pas.
        assert [ends for _, ends in segments] == [False] * (len(segments) - 1) + [True]

    def test_chaque_phrase_forme_son_segment(self):
        """C'est ce qui rend le point audible : un vrai silence, pas celui du modèle."""
        texte = " ".join(
            [
                "Le navire quitta le port au petit matin sous un ciel dégagé.",
                "Les marins hissèrent les voiles en chantant à pleine voix.",
                "La côte disparut lentement derrière eux dans la brume.",
            ]
        )
        segments = N.segment_paragraph(texte)
        assert len(segments) == 3
        assert all(ends for _, ends in segments)

    def test_phrase_trop_breve_rattachee_a_la_suivante(self):
        """Synthétisée seule, une phrase de dix caractères part en vrille."""
        texte = "Il partit. Le navire quitta le port au petit matin sous un ciel dégagé."
        segments = N.segment_paragraph(texte)
        assert len(segments) == 1
        assert segments[0][0].startswith("Il partit.")

    def test_le_seuil_ne_recolle_pas_des_phrases_ordinaires(self):
        courte = "Les marins hissèrent les voiles en chantant."
        assert len(courte) > N.MIN_SEGMENT_CHARS
        assert N.segment_paragraph(f"{courte} {courte}") == [(courte, True), (courte, True)]


class TestAnnonce:
    def test_titre_repris_du_numero_non_redit(self):
        """Un texte sans repère reçoit « Chapitre 1 » pour titre : ne pas le doubler."""
        assert N.announce(1, "Chapitre 1") == "Chapitre un."
        assert N.announce(3, "") == "Chapitre trois."

    def test_numero_discordant_ignore(self):
        """Les tables des matières se trompent : « Chapitre six. Chapitre sept. » non."""
        assert N.announce(6, "Chapitre 7") == "Chapitre six."

    def test_vrai_titre_annonce(self):
        assert N.announce(2, "Mort d'un personnage") == "Chapitre deux. Mort d'un personnage."


class TestSilences:
    def test_les_pauses_suivent_le_facteur(self):
        paragraphes = ["Le navire quitta le port au petit matin sous un ciel dégagé."]
        simple = N.build_chapter_segments(1, "Le départ", paragraphes)
        etire = N.build_chapter_segments(1, "Le départ", paragraphes, pause_scale=1.5)
        assert [r["pause_after_ms"] for r in etire] == [
            int(round(r["pause_after_ms"] * 1.5)) for r in simple
        ]


class TestReecriture:
    def test_un_contenu_identique_est_date_sans_etre_reecrit(self, tmp_path):
        """Repréparer date le fichier même sans changement : c'est la date de préparation,
        celle qui dit si le texte a été corrigé depuis. Le contenu, lui, ne bouge pas."""
        import os
        import time

        from voxlibris.normalize import build_segments

        text_dir, out_dir = tmp_path / "text", tmp_path / "segments"
        text_dir.mkdir()
        (text_dir / "ch01.md").write_text(
            "---\nchapter: 1\ntitle: Un\n---\n\nIl faisait beau ce matin-là.\n",
            encoding="utf-8",
        )
        build_segments(text_dir, out_dir)
        target = out_dir / "ch01.jsonl"
        before = target.read_bytes()
        ancien = time.time() - 3600
        os.utime(target, (ancien, ancien))

        build_segments(text_dir, out_dir)
        assert target.read_bytes() == before
        assert target.stat().st_mtime > ancien + 1

        (text_dir / "ch01.md").write_text(
            "---\nchapter: 1\ntitle: Un\n---\n\nIl pleuvait ce matin-là.\n",
            encoding="utf-8",
        )
        build_segments(text_dir, out_dir)
        assert target.read_bytes() != before
