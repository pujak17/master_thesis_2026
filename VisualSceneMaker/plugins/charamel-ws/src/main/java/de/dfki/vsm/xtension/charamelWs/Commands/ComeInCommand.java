package de.dfki.vsm.xtension.charamelWs.Commands;

public class ComeInCommand extends ActionCommand {

    public ComeInCommand(Direction direction) {
        super("humanoid/walk/come_in/", direction == Direction.RIGHT ? "come_in_right04.glb" : "come_in_left04.glb");
    }
}
