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
        for segment in N.segment_paragraph(N.normalize(paragraphe)):
            assert len(segment) <= N.MAX_CHARS

    def test_phrase_courte_non_decoupee(self):
        texte = "Une phrase unique et brève."
        assert N.segment_paragraph(texte) == [texte]

    def test_coupure_preferee_a_la_ponctuation_faible(self):
        # Une phrase trop longue se coupe sur ses virgules, pas au milieu des mots.
        longue = ", ".join(["un membre de phrase assez long pour compter"] * 8) + "."
        segments = N.segment_paragraph(longue)
        assert len(segments) > 1
        assert all(len(s) <= N.MAX_CHARS for s in segments)
        assert all(not s.endswith(" ") for s in segments)
