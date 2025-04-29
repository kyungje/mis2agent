import pytest
from src.ui.example import ExampleUI, UIComponent

@pytest.fixture
def ui():
    return ExampleUI()

@pytest.fixture
def component():
    return UIComponent(
        id="test-component",
        type="button",
        props={"text": "Click me", "color": "blue"}
    )

def test_add_component(ui, component):
    """컴포넌트 추가 테스트"""
    ui.add_component(component)
    assert len(ui.components) == 1
    assert ui.components[0].id == "test-component"

def test_get_component(ui, component):
    """컴포넌트 조회 테스트"""
    ui.add_component(component)
    found_component = ui.get_component("test-component")
    assert found_component == component
    assert found_component.type == "button"
    assert found_component.props["text"] == "Click me"

def test_get_nonexistent_component(ui):
    """존재하지 않는 컴포넌트 조회 테스트"""
    with pytest.raises(ValueError):
        ui.get_component("nonexistent")

def test_render(ui, component):
    """UI 렌더링 테스트"""
    ui.add_component(component)
    rendered = ui.render()
    assert "components" in rendered
    assert len(rendered["components"]) == 1
    assert rendered["components"][0]["id"] == "test-component"
    assert rendered["components"][0]["type"] == "button"
    assert rendered["components"][0]["props"]["text"] == "Click me" 