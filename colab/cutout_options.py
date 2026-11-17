"""Contratos comuns de seleção e comparação; não carregam modelos."""
import math


def validate_boxes(boxes):
    if not isinstance(boxes, list) or not 1 <= len(boxes) <= 20:
        raise ValueError('Finegrain precisa de uma seleção: clique em Selecionar objeto e arraste uma caixa ao redor da pessoa ou logo.')
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError('Região de seleção inválida.')
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in box):
            raise ValueError('Coordenadas da seleção precisam estar entre 0 e 1.')
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError('A seleção precisa ter largura e altura.')


def candidate_engines(options):
    # Presets explícitos; não inferir o tipo de arte a partir de um nome de arquivo.
    return {'logo': ('fey', 'ben2'), 'person': ('fey', 'birefnet-hr')}.get(options.get('cutoutSubject'), ('ben2', 'birefnet-hr'))
